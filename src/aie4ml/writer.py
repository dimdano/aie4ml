from hls4ml.writer.writers import Writer


class AIEWriter(Writer):
    """Placeholder AIE writer for the minimal aie4ml package."""

    def __init__(self):
        super().__init__()

    def write_hls(self, model):
        raise RuntimeError('Wait for full aie4ml release to enable AIE code generation.')
