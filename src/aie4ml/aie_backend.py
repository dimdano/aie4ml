import logging

from hls4ml.backends.backend import Backend
from hls4ml.model.optimizer import model_optimizer
from hls4ml.writer import get_writer

log = logging.getLogger(__name__)


class AIEBackend(Backend):
    """Placeholder AIE backend for the minimal aie4ml package."""

    def __init__(self):
        super().__init__('AIE')
        self.writer = get_writer(self.name)
        self._default_flow = None
        self._writer_flow = None

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def write(self, model):
        if self.writer is None:
            raise RuntimeError('No writer registered for the AIE backend in this minimal aie4ml package.')
        self.writer.write_hls(model)

    @model_optimizer()
    def write_hls(self, model):
        self.write(model)
        return True

    def compile(self, model):
        raise RuntimeError('Not yet implemented.')

    def predict(self, model, X):
        raise RuntimeError('Not yet implemented.')

    def create_initial_config(
        self,
        part='xilinx_vek280_base_202510_1',
        plio_width_bits=None,
        pl_clock_freq_mhz=None,
        iterations=8,
        namespace=None,
        write_tar=False,
        **_,
    ):
        pl_freq = pl_clock_freq_mhz if pl_clock_freq_mhz is not None else 312.5
        plio_width = plio_width_bits if plio_width_bits is not None else 128

        config = {
            'Part': part,
            'AIEConfig': {
                'Generation': 'AIE-ML',
                'PLIOWidthBits': plio_width,
                'PLClockFreqMHz': pl_freq,
                'Iterations': iterations,
            },
            'HLSConfig': {},
            'WriterConfig': {
                'Namespace': namespace,
                'WriteTar': write_tar,
            },
        }

        return config
