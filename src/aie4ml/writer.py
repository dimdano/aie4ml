# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shutil import copyfile

from hls4ml.writer.writers import Writer
from jinja2 import Environment, FileSystemLoader

from .ir import get_backend_context
from .passes.utils import sanitize_identifier
from .serialization import dump_pipeline_ir

config_filename = 'hls4ml_config.yml'


class AIEWriter(Writer):
    def __init__(self):
        super().__init__()
        self._template_root = Path(__file__).resolve().parent / 'templates'

    def write_aie(self, model):
        output_dir = Path(model.config.get_output_dir())
        self._prepare_directories(output_dir)

        ctx = get_backend_context(model)
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
        self._copy_kernel_sources(output_dir, model)
        self._render_templates(output_dir, model, layers, graph_plan, env)

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
            inst = ctx.ir.kernels.get(node.name)
            if inst is None:
                continue

            kernel_cfg = inst.config
            variant = inst.variant

            if node.name not in placements:
                raise RuntimeError(f'{inst.name}: missing physical placement; run placement pass before writer.')
            placement = dict(placements[node.name])

            layer_index += 1

            get_artifacts = getattr(variant, 'get_artifacts', None)
            artifacts = get_artifacts(inst) if get_artifacts is not None else []

            entry = {
                'index': layer_index,
                'inst_name': inst.name,
                'kernel_name': sanitize_identifier(node.name),
                'struct_name': f'L{layer_index}Cfg',
                'kernel': kernel_cfg,
                'placement': placement,
                'artifacts': artifacts,
            }

            # Optional metadata passthrough
            entry.update({k: node.metadata[k] for k in ('n_in', 'n_out', 'use_bias') if k in node.metadata})
            layers.append(entry)

        return layers

    def _emit_kernel_artifacts(self, output_dir: Path, layers, env):
        weights_dir = output_dir / 'src' / 'weights'
        for L in layers:
            for spec in L.get('artifacts', ()):
                if spec.get('array') is None:
                    continue

                tpl = env.get_template('artifacts_2d.h.jinja' if spec['kind'] == '2d' else 'artifacts_1d.h.jinja')
                out = weights_dir / spec['filename']
                out.write_text(
                    tpl.render(
                        inst_name=L['inst_name'],
                        artifact_name=spec['name'],
                        data=spec['array'],
                        dtype=spec['dtype'],
                    )
                )

    def _copy_kernel_sources(self, output_dir: Path, model):
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

        for dst, src in model.config.backend.get_custom_source().items():
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

    def _render_templates(self, output_dir: Path, model, layers, graph_plan, env: Environment):
        aie_cfg = model.config.get_config_value('AIEConfig', {})
        platform = model.config.get_config_value('Part')

        context = {
            'layers': layers,
            'graph_plan': graph_plan,
            'plio_bitwidth': int(aie_cfg['PLIOWidthBits']),
            'iterations': int(aie_cfg['Iterations']),
            'pl_freq_mhz': float(aie_cfg['PLClockFreqMHz']),
            'stamp': model.config.get_config_value('Stamp'),
            'platform': platform,
        }

        self._render_template(env, 'Makefile.jinja', output_dir / 'Makefile', context)
        self._render_template(env, 'aie.cfg.jinja', output_dir / 'aie.cfg', context)
        self._render_template(env, 'graph_plan.h.jinja', output_dir / 'src' / 'graph_plan.h', context)
        self._render_template(env, 'parameters.h.jinja', output_dir / 'src' / 'parameters.h', context)
        self._render_template(env, 'top_graph.h.jinja', output_dir / 'src' / 'top_graph.h', context)
        self._render_template(env, 'app.cpp.jinja', output_dir / 'app.cpp', context)

    def _render_template(self, env: Environment, template_name: str, destination: Path, context: dict):
        template = env.get_template(template_name)
        destination.write_text(template.render(**context))
