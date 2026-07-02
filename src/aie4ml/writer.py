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

        self._render_aie_templates(output_dir, ctx, layers, graph_plan, env)

        # Three rendering concerns: AIE compute sources (above), the PL+host system
        # (target='hardware' only), and the build Makefile (always; hardware-aware).
        # The system render owns and returns the pl_plan so the Makefile can reuse it
        # (build_pl_plan runs once). pl_plan is None for aie-only (no PL/host).
        from .system_plan import emits_system

        if emits_system(ctx):
            pl_plan = self._render_system_templates(output_dir, ctx, env)
        else:
            pl_plan = None
        self._render_makefile(output_dir, ctx, env, pl_plan)

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

    def _render_aie_templates(self, output_dir: Path, ctx, layers, graph_plan, env: Environment):
        """Render the AIE compute sources (always, hardware-agnostic)."""
        context = {
            'layers': layers,
            'graph_plan': graph_plan,
            'plio_bitwidth': ctx.device.plio_width_bits,
            'iterations': int(ctx.aie_config['Iterations']),
            'pl_freq_mhz': float(ctx.aie_config['PLClockFreqMHz']),
        }
        self._render_template(env, 'aie.cfg.jinja', output_dir / 'aie.cfg', context)
        self._render_template(env, 'graph_plan.h.jinja', output_dir / 'src' / 'graph_plan.h', context)
        self._render_template(env, 'parameters.h.jinja', output_dir / 'src' / 'parameters.h', context)
        self._render_template(env, 'top_graph.h.jinja', output_dir / 'src' / 'top_graph.h', context)
        self._render_template(env, 'app.cpp.jinja', output_dir / 'app.cpp', context)

    def _render_makefile(self, output_dir: Path, ctx, env: Environment, pl_plan):
        """Render the unified build Makefile (always). It is the one base-project template
        that is hardware-aware: when target='hardware' it adds PL/host/package targets and
        needs the data-mover plan (kernel .xo list, per-kernel build/clean). All such uses
        are gated behind is_hardware, so aie-only projects pass pl_plan=None.
        """
        context = {
            'project_name': ctx.project_config.project_name,
            'platform': ctx.device.platform,
            'stamp': ctx.project_config.stamp,
            'is_hardware': pl_plan is not None,
            'pl_plan': pl_plan,
        }
        self._render_template(env, 'Makefile.jinja', output_dir / 'Makefile', context)

    def _render_system_templates(self, output_dir: Path, ctx, env: Environment):
        """Render the PL data mover + v++ connectivity + XRT host (target='hardware').

        Templates live under templates/firmware/ (pl/, host/, system.cfg.jinja) and are
        rendered with the shared firmware Jinja environment. Returns the resolved
        ``pl_plan`` so the Makefile render can reuse it (built once, here).
        """
        from .system_plan import build_system_io

        system_io = build_system_io(ctx)

        (output_dir / 'pl').mkdir(parents=True, exist_ok=True)
        (output_dir / 'host').mkdir(parents=True, exist_ok=True)

        # Render each PL kernel the data-mover plan selects (benchmark = one combined
        # CU; memory_stream = mm2s + s2mm [+ tick_gen when timing]). Templates may live
        # under pl/ or pl/deployment/, but always render to pl/<name>.{cpp,cfg}.
        for kernel in system_io['pl_plan']['kernels']:
            name = kernel['name']
            self._render_template(env, kernel['cpp_template'], output_dir / 'pl' / f'{name}.cpp', system_io)
            self._render_template(env, kernel['cfg_template'], output_dir / 'pl' / f'{name}.cfg', system_io)

        self._render_template(env, 'system.cfg.jinja', output_dir / 'system.cfg', system_io)
        self._render_template(env, 'host/host.cpp.jinja', output_dir / 'host' / 'host.cpp', system_io)

        # DDR-packed input header (data.h) consumed by host.cpp.
        from .system_plan import host_data_context

        self._render_template(env, 'host/data.h.jinja', output_dir / 'host' / 'data.h', host_data_context(ctx))

        return system_io['pl_plan']

    def _render_template(self, env: Environment, template_name: str, destination: Path, context: dict):
        template = env.get_template(template_name)
        destination.write_text(template.render(**context))
