# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shutil import copyfile

from jinja2 import Environment, FileSystemLoader

from .passes.utils import sanitize_identifier
from .serialization import dump_pipeline_ir


class AIEProjectEmitter:
    """Framework-agnostic project emitter. Takes a populated AIEBackendContext and writes all output files."""

    def __init__(self):
        self._template_root = Path(__file__).resolve().parent / 'templates'

    def emit(self, ctx):
        output_dir = ctx.project_config.output_dir
        self._prepare_directories(output_dir)

        layers = self._collect_layers(ctx)
        graph_plan = ctx.ir.physical.plan or {}
        dump_pipeline_ir(ctx, output_dir / 'aie_pipeline.json')

        firmware_dir = self._template_root / 'firmware'
        env = Environment(
            loader=FileSystemLoader(str(firmware_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._emit_kernel_artifacts(output_dir, layers, env)
        self._copy_kernel_sources(output_dir, ctx.project_config.custom_sources)
        self._render_templates(output_dir, ctx, layers, graph_plan, env)

        # PL data mover + XRT host are emitted only for target='hardware' (default 'aie' is
        # AIE-only). Gated so the existing AIE/sim flow is completely unaffected.
        from .system_plan import emits_system

        if emits_system(ctx):
            self._render_system_templates(output_dir, ctx, env)

    def _prepare_directories(self, output_dir: Path):
        (output_dir / 'src').mkdir(parents=True, exist_ok=True)
        (output_dir / 'src' / 'kernels').mkdir(exist_ok=True)
        (output_dir / 'src' / 'weights').mkdir(exist_ok=True)
        (output_dir / 'data').mkdir(exist_ok=True)

    def _collect_layers(self, ctx):
        layers = []
        layer_index = 0
        placements = ctx.ir.physical.placements or {}

        for node in ctx.ir.logical:
            inst = ctx.ir.execution.get(node.name)
            if inst is None:
                continue

            variant = inst.variant

            if node.name not in placements:
                raise RuntimeError(f'{inst.name}: missing physical placement; run placement pass before writer.')
            placement = dict(placements[node.name])

            layer_index += 1
            artifacts = variant.get_artifacts(inst)

            sanitized_name = sanitize_identifier(inst.name)
            entry = {
                'index': layer_index,
                'inst_name': sanitized_name,
                'op_impl_name': sanitized_name,
                'struct_name': f'L{layer_index}Cfg',
                'op_impl': {
                    'graph_header': inst.graph_header,
                    'graph_name': inst.graph_name,
                    'param_template': inst.param_template,
                    'parameters': variant.build_template_params(node, inst.config),
                },
                'io_views': inst.io_views,
                'placement': placement,
                'artifacts': artifacts,
            }
            entry.update({k: node.metadata[k] for k in ('n_in', 'n_out') if k in node.metadata})
            layers.append(entry)

        return layers

    def _emit_kernel_artifacts(self, output_dir: Path, layers, env):
        weights_dir = output_dir / 'src' / 'weights'
        for L in layers:
            for spec in L.get('artifacts', ()):
                if 'storage' not in spec:
                    raise RuntimeError(f"{L['inst_name']}: artifact {spec.get('name')} missing storage metadata.")
                if spec.get('array') is None:
                    continue

                tpl = env.get_template('artifacts_2d.h.jinja' if spec['kind'] == '2d' else 'artifacts_1d.h.jinja')
                out = weights_dir / spec['filename']
                out.write_text(
                    tpl.render(
                        inst_name=L['inst_name'],
                        artifact_name=spec['name'],
                        data=spec['array'],
                        dtype=spec.get('storage_dtype', spec['dtype']),
                    )
                )

    def _copy_kernel_sources(self, output_dir: Path, custom_sources: dict):
        src_kernel_dir = self._template_root / 'nnet_utils'
        dst_kernel_dir = output_dir / 'src' / 'kernels'

        if dst_kernel_dir.exists():
            for p in dst_kernel_dir.iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    self._remove_tree(p)

        dst_kernel_dir.mkdir(exist_ok=True)

        for src in src_kernel_dir.rglob('*'):
            if src.is_file():
                dst = dst_kernel_dir / src.relative_to(src_kernel_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                copyfile(src, dst)

        for dst, src in custom_sources.items():
            dst = output_dir / dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            copyfile(src, dst)

    def _remove_tree(self, path: Path):
        if path.is_dir():
            for c in path.iterdir():
                self._remove_tree(c)
            path.rmdir()
        else:
            path.unlink()

    def _render_templates(self, output_dir: Path, ctx, layers, graph_plan, env: Environment):
        from .system_plan import emits_system

        context = {
            'layers': layers,
            'graph_plan': graph_plan,
            'plio_bitwidth': ctx.device.plio_width_bits,
            'iterations': int(ctx.aie_config['Iterations']),
            'pl_freq_mhz': float(ctx.aie_config['PLClockFreqMHz']),
            'stamp': ctx.project_config.stamp,
            'platform': ctx.device.platform,
            # The unified Makefile adds PL/host/package targets when this project was
            # emitted for hardware; aie-only projects render just the AIE/sim flow.
            'is_hardware': emits_system(ctx),
            'project_name': ctx.project_config.project_name,
        }

        self._render_template(env, 'Makefile.jinja', output_dir / 'Makefile', context)
        self._render_template(env, 'aie.cfg.jinja', output_dir / 'aie.cfg', context)
        self._render_template(env, 'graph_plan.h.jinja', output_dir / 'src' / 'graph_plan.h', context)
        self._render_template(env, 'parameters.h.jinja', output_dir / 'src' / 'parameters.h', context)
        self._render_template(env, 'top_graph.h.jinja', output_dir / 'src' / 'top_graph.h', context)
        self._render_template(env, 'app.cpp.jinja', output_dir / 'app.cpp', context)

    def _render_system_templates(self, output_dir: Path, ctx, env: Environment):
        """Render the PL data mover + v++ connectivity + XRT host (target='hardware').

        Templates live under templates/firmware/ (pl/, host/, system.cfg.jinja) and are
        rendered with the shared firmware Jinja environment.
        """
        from .system_plan import build_system_io

        system_io = build_system_io(ctx)

        (output_dir / 'pl').mkdir(parents=True, exist_ok=True)
        (output_dir / 'host').mkdir(parents=True, exist_ok=True)

        self._render_template(
            env, 'pl/ddr_pl_aie_datamover.cpp.jinja', output_dir / 'pl' / 'ddr_pl_aie_datamover.cpp', system_io
        )
        self._render_template(
            env, 'pl/ddr_pl_aie_datamover.cfg.jinja', output_dir / 'pl' / 'ddr_pl_aie_datamover.cfg', system_io
        )
        self._render_template(env, 'system.cfg.jinja', output_dir / 'system.cfg', system_io)
        self._render_template(env, 'host/host.cpp.jinja', output_dir / 'host' / 'host.cpp', system_io)

        # DDR-packed input header (data.h) consumed by host.cpp.
        from .system_plan import write_host_data_header

        write_host_data_header(ctx, output_dir)

    def _render_template(self, env: Environment, template_name: str, destination: Path, context: dict):
        template = env.get_template(template_name)
        destination.write_text(template.render(**context))
