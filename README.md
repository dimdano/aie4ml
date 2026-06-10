<p align="center">
  <img src="https://github.com/dimdano/aie4ml/blob/main/docs/aie4ml_logo_big.png" alt="aie4ml" width="600"/>
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.15946-b31b1b.svg)](https://arxiv.org/abs/2512.15946)

`aie4ml` is an end-to-end compiler that generates **optimized** AIE firmware automatically, which can be then built and simulated directly using **AMD Vitis**. It targets the **AMD AI Engine (AIE)** from model-level frontends and lowers supported operators into AIE graphs and kernels as a standalone AIE project.

- Current hardware targets: AIE-ML and AIE-MLv2 devices.
- Current frontend paths: ONNX for explicit operator graphs, and an optional [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) frontend path.

## Current Support

aie4ml currently supports Dense/GEMM, dynamic MatMul, Elementwise Add, quantized LayerNorm, HCCS Softmax, final-two-axis
Permute, Split/Slice, Concat, fanout, and fused ReLU across AIE-ML and AIE-MLv2 devices.

See the [operator and feature support matrix](docs/support.md) for supported precisions, frontend coverage, tensor and
transport contracts, and current limitations.

## Prerequisites

- AMD Vitis 2025.2 and a valid AIE tools license.
- Python 3.10+.
- Optional: [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) if using the hls4ml frontend integration.

## Frontend Compatibility

The ONNX path is the recommended route for operator-level compiler development and for models that already express quantized tensors and Q/DQ boundaries explicitly. The hls4ml path is intended for MLP-style pipelines at the moment.

## Installation

```bash
pip install aie4ml
```

Install hls4ml only if you need the hls4ml frontend/backend integration:

```bash
pip install hls4ml
```

## Documentation & Tutorials

Documentation and usage: [https://github.com/dimdano/aie4ml](https://github.com/dimdano/aie4ml)

Tutorial 1: [`tutorials/tutorial_1.ipynb`](tutorials/tutorial_1.ipynb)
Tutorial 2: [`tutorials/tutorial_2.ipynb`](tutorials/tutorial_2.ipynb)

General `hls4ml` concepts: [https://fastmachinelearning.org/hls4ml](https://fastmachinelearning.org/hls4ml)


## Maintainer

`aie4ml` is developed and maintained by [Dimitrios Danopoulos](https://github.com/dimdano).

## Citation

If `aie4ml` contributes to your research, please cite the corresponding arXiv preprint:

```bibtex
@misc{danopoulos2025aie4mlendtoendframeworkcompiling,
      title={AIE4ML: An End-to-End Framework for Compiling Neural Networks for the Next Generation of AMD AI Engines},
      author={Dimitrios Danopoulos and Enrico Lupi and Chang Sun and Sebastian Dittmeier and Michael Kagan and Vladimir Loncar and Maurizio Pierini},
      year={2025},
      eprint={2512.15946},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.15946},
}
```
