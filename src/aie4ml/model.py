# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from .ir import PhysicalIR, get_backend_context
from .ir.context import CONTEXT_ATTR, AIEBackendContext
from .passes import run_aie_passes
from .pipeline import DEFAULT_PIPELINE
from .simulation import (
    build_io_layout,
    collect_outputs,
    dequantize_outputs,
    prepare_inputs,
    run_simulation_target,
    write_input_files,
)
from .writer import AIEProjectEmitter

log = logging.getLogger(__name__)


class AIEModel:
    """Backend-owned runtime/project object shared by all frontends."""

    def __init__(self, ctx: AIEBackendContext, source_model: Optional[Any] = None):
        setattr(self, CONTEXT_ATTR, ctx)
        self.source_model = source_model

    @classmethod
    def from_context(cls, ctx: AIEBackendContext, source_model: Optional[Any] = None) -> 'AIEModel':
        return cls(ctx, source_model=source_model)

    @property
    def context(self) -> AIEBackendContext:
        return get_backend_context(self)

    def run_pipeline(self) -> 'AIEModel':
        run_aie_passes(self.context, [cls() for cls in DEFAULT_PIPELINE])
        return self

    def _pipeline_json_path(self) -> Path:
        return self.context.project_config.output_dir / 'aie_pipeline.json'

    def _load_emitted_physical_plan(self) -> None:
        ctx = self.context
        source = self._pipeline_json_path()
        if not source.exists():
            raise FileNotFoundError(
                f'Project directory "{ctx.project_config.output_dir}" exists but "{source.name}" is missing.'
            )

        data = json.loads(source.read_text())
        try:
            ctx.ir.physical = PhysicalIR.from_dict(data.get('physical'), require_plan_buffers=True)
        except RuntimeError as exc:
            raise RuntimeError(f'"{source}": {exc}') from exc

    def _ensure_runtime_plan(self) -> None:
        ctx = self.context
        if ctx.ir.physical.plan and ctx.ir.physical.placements:
            return

        output_dir = ctx.project_config.output_dir
        if not output_dir.exists():
            self.run_pipeline()
            return

        self._load_emitted_physical_plan()

    def write(self) -> 'AIEModel':
        self.run_pipeline()
        AIEProjectEmitter().emit(self.context)
        return self

    def build(self, make_target: str = 'all', env=None, log_to_stdout: bool = True) -> int:
        ctx = self.context
        output_dir = ctx.project_config.output_dir
        self._ensure_runtime_plan()
        if not output_dir.exists():
            self.write()

        cmd = ['make', make_target]
        log.debug('Running %s in %s', ' '.join(cmd), output_dir)

        stdout = None if log_to_stdout else subprocess.PIPE
        stderr = None if log_to_stdout else subprocess.STDOUT
        result = subprocess.run(cmd, cwd=output_dir, env=env, stdout=stdout, stderr=stderr, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Make target "{make_target}" failed for project "{ctx.project_config.project_name}"')
        if not log_to_stdout and result.stdout:
            log.info(result.stdout)
        return result.returncode

    def compile(self) -> int:
        return self.build(make_target='x86com')

    def predict(
        self,
        X,
        simulator: str = 'x86',
        *,
        quantize_in: bool = True,
        dequantize_out: bool = True,
    ):
        ctx = self.context
        output_dir = ctx.project_config.output_dir
        if not output_dir.exists():
            raise FileNotFoundError(
                f'Output directory "{output_dir}" does not exist. Run write() and compile() before predicting.'
            )

        self._ensure_runtime_plan()
        layout = build_io_layout(self)
        iterations = int(ctx.aie_config['Iterations'])
        plio_width = ctx.device.plio_width_bits
        prepared_inputs = prepare_inputs(layout, X, iterations=iterations, quantize=quantize_in)
        write_input_files(output_dir, layout, prepared_inputs, plio_width_bits=plio_width)

        sim_key = simulator.lower()
        if sim_key == 'x86':
            make_target = 'x86sim'
        elif sim_key == 'aie':
            make_target = 'profile'
        else:
            raise ValueError(f'Unknown simulator "{simulator}". Expected one of: x86, aie.')

        log.info('Running %s simulation using make %s', ctx.project_config.project_name, make_target)
        run_simulation_target(output_dir, make_target)

        sim_out = collect_outputs(output_dir, sim_key, layout)
        final_out = dequantize_outputs(layout, sim_out) if dequantize_out else sim_out

        def _flatten_iters(arr):
            if getattr(arr, 'ndim', 0) >= 2:
                return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
            return arr

        if len(final_out) == 1:
            return _flatten_iters(next(iter(final_out.values())))
        return {k: _flatten_iters(v) for k, v in final_out.items()}

    def report(self):
        """Resource, latency and per-kernel cycle report for this project."""
        from .report import report

        return report(self)


def from_hls4ml(model) -> AIEModel:
    if not hasattr(model, CONTEXT_ATTR):
        from .frontends.hls4ml.lower import LowerToAieIr

        LowerToAieIr().transform(model)
    ctx = get_backend_context(model)
    return AIEModel.from_context(ctx, source_model=model)
