# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import subprocess
from pathlib import Path

from hls4ml.backends.backend import Backend
from hls4ml.backends.fpga.fpga_backend import FPGABackend as _FPGABackendHelper
from hls4ml.model.attributes import Attribute, ConfigurableAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import Activation, Dense
from hls4ml.model.optimizer import layer_optimizer, model_optimizer
from hls4ml.writer import get_writer

from .device_catalog import load_device_catalog

log = logging.getLogger(__name__)


class AIEBackend(Backend):
    def __init__(self):
        super().__init__('AIE')
        self.writer = get_writer(self.name)
        self.attribute_map = {}
        self._register_aie_layer_attributes()
        self._register_flows()

    def _register_aie_layer_attributes(self):
        dense_attrs = self.attribute_map.get(Dense, [])
        custom_dense_attrs = [
            ConfigurableAttribute('cas_num', default=-1),
            ConfigurableAttribute('cas_length', default=-1),
            ConfigurableAttribute('tiling', value_type=dict, default={}),
            Attribute('placement', value_type=dict, default={}, configurable=True),
        ]
        for attr in custom_dense_attrs:
            if attr not in dense_attrs:
                dense_attrs.append(attr)
        self.attribute_map[Dense] = dense_attrs

        activation_attrs = self.attribute_map.get(Activation, [])
        custom_act_attrs = [
            Attribute('placement', value_type=dict, default={}, configurable=True),
        ]
        for attr in custom_act_attrs:
            if attr not in activation_attrs:
                activation_attrs.append(attr)
        self.attribute_map[Activation] = activation_attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)
        lower_flow = register_flow('lower', ['aie:lower_to_aie_ir'], requires=[init_flow], backend=self.name)
        quant_flow = register_flow('quantize', ['aie:integer_quantizer'], requires=[lower_flow], backend=self.name)
        fuse_flow = register_flow('fuse', ['aie:fuse_activation_casts'], requires=[quant_flow], backend=self.name)
        resolve_flow = register_flow('resolve', ['aie:resolve'], requires=[fuse_flow], backend=self.name)
        pack_flow = register_flow('pack', ['aie:pack_kernel_artifacts'], requires=[resolve_flow], backend=self.name)
        placement_flow = register_flow('placement', ['aie:place_kernels'], requires=[pack_flow], backend=self.name)
        memory_plan_flow = register_flow(
            'memory_plan', ['aie:build_memory_plan'], requires=[placement_flow], backend=self.name
        )
        template_flow = register_flow(
            'apply_templates', self._get_layer_templates, requires=[memory_plan_flow], backend=self.name
        )

        self._default_flow = register_flow('project', None, requires=[template_flow], backend=self.name)
        writer_passes = ['make_stamp', 'aie:write_aie']
        self._writer_flow = register_flow('write', writer_passes, requires=[self._default_flow], backend=self.name)

    def create_layer_class(self, layer_class):
        new_attributes = []
        for cls, attributes in self.attribute_map.items():
            if issubclass(layer_class, cls):
                new_attributes.extend(attributes)

        layer_cls_fqn = layer_class.__module__ + '.' + layer_class.__qualname__

        return type(
            self.name + layer_class.__name__,
            (layer_class,),
            {'_expected_attributes': new_attributes, '_wrapped': layer_cls_fqn},
        )

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    @layer_optimizer(Dense)
    def init_dense_defaults(self, layer):
        if layer.get_attr('tiling', None) is None:
            layer.set_attr('tiling', {})
        if layer.get_attr('placement', None) is None:
            layer.set_attr('placement', {})

    @layer_optimizer(Activation)
    def init_activation_defaults(self, layer):
        pass

    def compile(self, model):
        """Compile the generated project for x86 simulation.

        The AIE flow relies on the generated Makefile targets. Invoking the
        ``x86com`` target builds the host application and graph for execution
        on the x86 functional simulator.
        """

        log.info('Compiling %s using make x86com', model.config.get_project_name())
        self.build(model, make_target='x86com')

    def predict(self, model, X, simulator='x86', quantize_io=True):
        from .simulation import (
            collect_outputs,
            dequantize_outputs,
            extract_dense_io,
            numpy_emulate,
            prepare_input_tensor,
            run_simulation_target,
            write_input_files,
            write_numpy_outputs,
        )

        output_dir = Path(model.config.get_output_dir())
        if not output_dir.exists():
            raise FileNotFoundError(
                f'Output directory "{output_dir}" does not exist. Run write() and compile() before predicting.'
            )

        first_layer, last_layer = extract_dense_io(model)
        aie_cfg = model.config.get_config_value('AIEConfig', {}) or {}
        batch_size = int(aie_cfg.get('BatchSize'))
        iterations = int(aie_cfg.get('Iterations'))
        quantize_inputs = bool(quantize_io)

        prepared_inputs = prepare_input_tensor(
            model=model,
            X=X,
            io_info=first_layer,
            batch_size=batch_size,
            iterations=iterations,
            quantize_inputs=quantize_inputs,
        )

        plio_width = int(aie_cfg.get('PLIOWidthBits', 128))
        vals_per_line_in = max(1, plio_width // first_layer.input_bitwidth)
        write_input_files(output_dir, prepared_inputs, first_layer, vals_per_line_in)

        sim_key = simulator.lower()
        if sim_key == 'x86':
            make_target = 'x86sim'
        elif sim_key == 'aie':
            make_target = 'aiesim'
        else:
            raise ValueError(f'Unknown simulator "{simulator}". Expected one of: x86, aie.')

        log.info('Running %s simulation using make %s', model.config.get_project_name(), make_target)
        run_simulation_target(output_dir, make_target)

        # numpy sim needed only for debugging purposes
        np_out = numpy_emulate(model, prepared_inputs)
        vals_per_line_out = max(1, plio_width // last_layer.output_bitwidth)
        write_numpy_outputs(
            output_dir=output_dir,
            np_outputs=np_out,
            io_info=last_layer,
            batch_size=batch_size,
            iterations=iterations,
            vals_per_line=vals_per_line_out,
        )

        sim_out = collect_outputs(output_dir, sim_key, last_layer, batch_size, iterations)
        final_out = dequantize_outputs(sim_out, last_layer) if quantize_inputs else sim_out

        return final_out

    def write(self, model):
        model.apply_flow(self.get_writer_flow())

    @classmethod
    def convert_precision_string(cls, precision):
        return _FPGABackendHelper.convert_precision_string(precision)

    def _get_device_info(self, part):
        catalog = load_device_catalog()
        if part is None:
            available = ', '.join(sorted(catalog)) or '<none>'
            raise ValueError(f'No AIE part specified. Available catalog entries: {available}.')
        try:
            return catalog[part]
        except KeyError as exc:
            available = ', '.join(sorted(catalog)) or '<none>'
            raise ValueError(f'Unknown part "{part}". Available catalog entries: {available}.') from exc

    def create_initial_config(
        self,
        part='xilinx_vek280_base_202520_1',
        plio_width_bits=None,
        pl_clock_freq_mhz=None,
        batch_size=8,
        iterations=8,
        column_start=None,
        row_start=None,
        namespace=None,
        write_tar=False,
        **_,
    ):
        device_info = copy.deepcopy(self._get_device_info(part))

        def _require(key):
            if key not in device_info:
                raise KeyError(f'Device catalog entry "{part}" missing required key "{key}".')
            return device_info[key]

        plio_width = plio_width_bits if plio_width_bits is not None else _require('PLIOWidthBits')
        pl_freq = pl_clock_freq_mhz if pl_clock_freq_mhz is not None else _require('PLClockFreqMHz')
        col_start_val = column_start if column_start is not None else _require('ColumnStart')
        row_start_val = row_start if row_start is not None else _require('RowStart')
        if 'MaxMemTileInPorts' not in device_info or 'MaxMemTileOutPorts' not in device_info:
            raise KeyError(f'Device catalog entry "{part}" missing MaxMemTile port limits.')

        config = {
            'Part': part,
            'AIEConfig': {
                'Device': _require('DeviceName'),
                'Generation': _require('Generation'),
                'Columns': _require('Columns'),
                'Rows': _require('Rows'),
                'ColumnStart': col_start_val,
                'RowStart': row_start_val,
                'PLIOWidthBits': plio_width,
                'PLClockFreqMHz': pl_freq,
                'BatchSize': batch_size,
                'Iterations': iterations,
                'Memory': device_info.get('Memory'),
                'MaxMemTileInPorts': int(device_info['MaxMemTileInPorts']),
                'MaxMemTileOutPorts': int(device_info['MaxMemTileOutPorts']),
            },
            'HLSConfig': {},
            'WriterConfig': {
                'Namespace': namespace,
                'WriteTar': write_tar,
            },
        }

        return config

    def build(self, model, make_target='all', env=None, log_to_stdout=True):
        output_dir = Path(model.config.get_output_dir())
        if not output_dir.exists():
            raise FileNotFoundError(f'Output directory "{output_dir}" does not exist. Run .write() first.')

        cmd = ['make', make_target]
        log.debug('Running %s in %s', ' '.join(cmd), output_dir)

        stdout = None if log_to_stdout else subprocess.PIPE
        stderr = None if log_to_stdout else subprocess.STDOUT

        result = subprocess.run(cmd, cwd=output_dir, env=env, stdout=stdout, stderr=stderr, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Make target "{make_target}" failed for project "{model.config.get_project_name()}"')

        if not log_to_stdout and result.stdout:
            log.info(result.stdout)

        return result.returncode

    @model_optimizer()
    def write_aie(self, model):
        self.writer.write_aie(model)
        return True
