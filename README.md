<p align="center">
  <img src="https://github.com/dimdano/aie4ml/blob/main/docs/aie4ml_logo_big.png" alt="aie4ml" width="600"/>
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.15946-b31b1b.svg)](https://arxiv.org/abs/2512.15946)

`aie4ml` is an end-to-end compiler that generates **optimized** AIE firmware automatically, which can be then built and simulated directly using **AMD Vitis**. It targets the **AMD AI Engine (AIE)** from model-level frontends and lowers supported operators into AIE graphs and kernels as a standalone AIE project.

- Current hardware targets: AIE1, AIE-ML and AIE-MLv2 devices.
- Current frontend paths: ONNX for explicit operator graphs, and an optional [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) frontend path.

## Current Support

The tables describe implemented compiler paths, not every combination of shape, precision and routing. A check mark means the feature is available subject to the limits in its row; unsupported combinations fail during lowering. See [detailed support and transport contracts](docs/support.md) for the full constraints.

### Compute layers

| Layer | AIE1 | AIE-ML | AIE-MLv2 | Precision and supported configuration | Parallelism | Data ports |
| --- | :---: | :---: | :---: | --- | --- | --- |
| Dense / static-weight GEMM | ✓ | ✓ | ✓ | Integer or floating-point formats below; constant weights, optional bias and fused ReLU. | `cas_num` splits output channels (`inner`) or rows (`outer`); `cas_length` splits the reduction. | Buffer or stream. |
| Dynamic MatMul | ✓ | ✓ | ✓ | Same format families as Dense; rank-2 RHS, optionally broadcast across LHS leading axes. Non-broadcast batched RHS is unsupported. | `inner` or `outer` `cas_num`, plus reduction `cas_length`. | Buffer. |
| Conv2D / grouped / depthwise | ✓ | ✓ | ✓ | Signed int8 input, weights and output; static weights, optional bias/ReLU, zero `pads`, `groups` and `kernel_shape` validated through 7×7. Batch > 1 and dilation are unsupported. Stride > 1 runs on one buffer-port tile, fed by the graph boundary or another kernel, through a retiler on the neighbouring tile. | Buffer: `inner` output-channel or `outer` row-band `cas_num`, and input-channel-block `cas_length`; splits must fit whole blocks/bands. Stream: one tile only. | Buffer or stream for stride 1; buffer only for stride > 1. |
| Elementwise Add | ✓ | ✓ | ✓ | Inputs and output must have the same shape and storage type; no broadcasting. | `inner` or `outer` `cas_num`; no cascade reduction. | Buffer. |
| LayerNorm | ✓ | ✓ | ✓ | Signed int8 input/output; `axis=-1`, constant gamma/beta, supported static quantization and representable positive `epsilon`. Linear and microtiled kernels. | `outer` `cas_num`; `cas_length=1`. | Buffer. |
| Softmax | ✓ | ✓ | ✓ | Int8 input to uint8 Q8 or int16 Q15; `axis=-1`. Accurate integer exponential or opt-in QAT surrogate (`approximation`); linear or microtiled `layout`. | `outer` `cas_num`; `cas_length=1`. | Buffer. |

Dense and MatMul input × weight formats registered by generation (output precision and scale must also pass the accumulator/shift checks):

| Format | AIE1 | AIE-ML | AIE-MLv2 |
| --- | :---: | :---: | :---: |
| int8 × int8, int16 × int8, float32 × float32 | ✓ | ✓ | ✓ |
| int16 × int16, bfloat16 × bfloat16 | — | ✓ | ✓ |
| FP8 E4M3 × FP8 E4M3 | — | — | ✓ |

Int8 × int16 is deliberately rejected on all generations because the current accumulator output shift may be negative. Exact microtile choices are generation-specific and validated when selected.

### Graph and transport features

| Feature | AIE1 | AIE-ML | AIE-MLv2 | Supported contract |
| --- | :---: | :---: | :---: | --- |
| Fused ReLU and constant scale | ✓ | ✓ | ✓ | ReLU folds into Dense or Conv2D; power-of-two scale folds into integer output shifts. No standalone activation or arbitrary-scale kernel. |
| Flatten / Reshape | ✓ | ✓ | ✓ | One sample to `[1, K]`, folded into a compatible producer (currently Conv2D) and consuming Dense; no copy kernel. |
| Transpose / Permute | ✓ | ✓ | ✓ | Final-two-axis view only. AIE1 accepts foldable views but rejects a relayout requiring a memory tile. |
| Split / Slice | ✓ | ✓ | ✓ | Folded, unit-step, port-aligned slices; no cross-port repacking or graph-boundary/chained views. |
| Concat | ✓ | ✓ | ✓ | Folded when each consumer port belongs to one input; no port-spanning gather or graph-boundary/chained views. |
| Fanout / branching | ✓ | ✓ | ✓ | Each consumer is planned separately; AIE1 direct multicast requires compatible staging. Routing and DMA resources still bound fanout. |
| Direct buffer connections | ✓ | ✓ | ✓ | Compatible staging is required; shared-neighbour-memory aliasing is a placement optimization, not a guarantee. |
| Memory-tile staging | — | ✓ | ✓ | One stage when supported by the device and layout; multi-stage relay and arbitrary relayout are not implemented. |

ONNX is the recommended frontend for supported quantized operator graphs with explicit Q/DQ boundaries. The optional hls4ml path supports MLP-style Dense stacks, Conv2D and DepthwiseConv2D; SeparableConv2D must be expressed as separate depthwise and pointwise layers. Hardware-specific placement, memory and DMA limits are detailed in [support.md](docs/support.md) rather than implied by a check mark.

## Prerequisites

- The latest AMD Vitis (currently 2026.1) and a valid AIE tools license. aie4ml tracks the newest AIE
  compiler; older releases can fail to compile some kernels.
- Python 3.10+.
- Optional: [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) if using the hls4ml frontend integration.

## Frontend Compatibility

The ONNX path is the recommended route for operator-level compiler development and for models that already express quantized tensors and Q/DQ boundaries explicitly. The hls4ml path supports Dense stacks and the documented Conv2D and DepthwiseConv2D configurations.

## Installation

```bash
pip install aie4ml
```

For the ONNX frontend (recommended path), also install ONNX and onnxruntime:

```bash
pip install onnx onnxruntime
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

If `aie4ml` contributes to your research, please cite the corresponding publications:

```bibtex
@INPROCEEDINGS{11552717,
  author={Danopoulos, Dimitrios and Lupi, Enrico and Sun, Chang and Dittmeier, Sebastian and Kagan, Michael and Loncar, Vladimir and Pierini, Maurizio},
  booktitle={2026 IEEE 34th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
  title={AIE4ML: An End-to-End Framework for Compiling Neural Networks for the Next Generation of AMD AI Engines},
  year={2026},
  volume={},
  number={},
  pages={176-184},
  keywords={Tiles;Modeling;Arrays;Kernel;Memory;Information rates;Throughput;System-on-chip;Loading;Engines;ai engines;hls4ml;aie4ml;versal;acceleration;inference},
  doi={10.1109/FCCM68464.2026.00035}}
```

```bibtex
@misc{danopoulos2026tamingexponentialfastsoftmax,
      title={Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference},
      author={Dimitrios Danopoulos and Enrico Lupi and Michael Kagan and Maurizio Pierini},
      year={2026},
      eprint={2604.02292},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.02292},
}
```
