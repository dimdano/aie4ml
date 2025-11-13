<p align="center">
  <img src="https://github.com/dimdano/aie4ml/blob/main/docs/aie4ml_logo_big.png" alt="aie4ml" width="600"/>
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)

`aie4ml` (to be published soon) is a plugin backend for [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) that targets the **AMD AI Engine (AIE)**.  `aie4ml` generates **ready-to-compile AIE projects** that can be built directly using **AMD Vitis**.

- Supports linear (dense) layers, with optional bias and ReLU activation
- Produces optimized implementations that can automatically scale across the AIE array
- Offers much shorter compilation/synthesis times compared to traditional FPGA HLS flows
- Enables high throughput especially for large models
- Currently supports Gen2 architecture devices (AIE-ML) with plans for broader support in the future.

If you have any questions, comments, or ideas regarding `aie4ml`, please open an issue or start a discussion in this repository.

# Documentation & Tutorials

Full documentation and usage notes will be maintained here:
👉 [https://github.com/dimdano/aie4ml](https://github.com/dimdano/aie4ml)

For general `hls4ml` concepts (model conversion, quantisation, configuration), see:
👉 [https://fastmachinelearning.org/hls4ml](https://fastmachinelearning.org/hls4ml)

# Installation

```bash
pip install .
