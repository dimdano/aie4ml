# Operator and Feature Support


## Operators

| Operator / feature | Current support | Notes and limitations |
| --- | --- | --- |
| Dense / GEMM | Int8/int16, BF16, FP8 | Read-only RTP weights; optional bias and fused ReLU; configurable cascade parallelism. |
| Dynamic MatMul | Int8/int16, BF16, FP8 | Rank-2 GEMM ABI. A rank-2 RHS may be broadcast across compacted independent LHS axes; batched RHS MatMul with non-broadcast leading axes is rejected. |
| Elementwise Add | Quantized integer | Exact-shape inputs for residual and elementwise connections; broadcasting is not supported. |
| LayerNorm | Quantized integer | Last-axis normalization using integer mean/variance and reciprocal-square-root approximation. Requires supported static quantization and kernel legality constraints. |
| Softmax (approx.) | Int8 to uint8/int16 | Calibrated clipped-linear Softmax surrogate, not exponential Softmax. Requires explicit parameters, offline calibration, and typically QAT. |
| Transpose / Permute | Memtile-backed view | Permutation of the final two axes only. |
| Split / Slice | Direct or per-slice memtile | No Split/Slice kernel. A slice must be an exact union of complete producer-port regions; crossing producer ports, graph-boundary slices, and chained views are not supported. |
| Concat | Direct or per-input memtile | No Concat kernel. Each consumer port must read entirely from one input slice; graph-input-backed Concat and chained views are not supported. |
| Fanout / branching | Direct or per-consumer memtile | One producer tensor may feed multiple independently planned consumer transport legs. |
| Constant scale | Quantized integer | Constant power-of-two output scaling is folded into Dense/MatMul output shifts. Arbitrary constant scaling is not yet supported. |
| Activation | ReLU | Fused into Dense; no standalone activation kernel. |

## Frontends

| Frontend | Current support | Notes and limitations |
| --- | --- | --- |
| ONNX | Recommended operator-level frontend | Supports explicit graphs composed from supported operators and quantized Q/DQ boundaries. |
| hls4ml | Optional MLP-oriented frontend | Intended primarily for Dense-style pipelines. Install `hls4ml` separately when using this path. |


## Tensor and View Contracts

- AIE execution buffers remain 2-D. Rank-preserving logical views are compacted only where doing so preserves operator
  semantics.
- Dynamic MatMul rejects a non-broadcast batched RHS. Such operations must be lowered into parallel rank-2 MatMul
  subgraphs or use a future batched-MatMul implementation.
- Permute supports the final two axes only.
- Split/Slice and Concat are folded view operations and do not instantiate AIE kernels.
- Chained folded views are not currently supported.
- Per-tensor static quantization is the primary supported quantization contract. Per-channel activation quantization is
  not supported.

## Transport

- Internal transport is realized as either a direct AIE connection or one memory-tile stage.
- Fanout creates independently planned transport legs for each consumer.
- Split/Slice and Concat may use direct connections when producer and consumer port regions align exactly; otherwise
  each legal leg may use a memory tile.
- Graph boundaries may expose multiple ports for partitioned tensors.
- Multi-stage relay transport is not implemented. Topologies requiring an additional relay stage fail explicitly.
- Internal AIE-to-PL-to-AIE bridge points and complete Versal system-link generation are not yet implemented.
