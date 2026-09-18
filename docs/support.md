# Operator and Feature Support


## Operators

| Operator / feature | Current support | Notes and limitations |
| --- | --- | --- |
| Dense / GEMM | Generation-dependent integer and float formats | Read-only RTP weights; optional bias and fused ReLU; configurable cascade parallelism. Exact precision/microtile combinations are device-specific. AIE-ML supports BF16 but not FP8; FP8 is available on AIE-MLv2. |
| Dynamic MatMul | Generation-dependent integer and float formats | Rank-2 GEMM ABI with device-specific precision/microtile combinations. A rank-2 RHS may be broadcast across compacted independent LHS axes; batched RHS MatMul with non-broadcast leading axes is rejected. FP8 is available on AIE-MLv2. |
| Elementwise Add | Generation-dependent integer and float formats | Exact-shape inputs for residual and elementwise connections; broadcasting is not supported. BF16 is supported on AIE-ML and FP8 on AIE-MLv2. |
| LayerNorm | Signed int8 | Last-axis normalization using integer mean/variance and reciprocal-square-root approximation. Input and output storage are currently signed int8 and require supported static quantization. |
| Softmax | Int8 to uint8/int16 | Accurate integer exponential or an opt-in surrogate (needs explicit parameters + QAT). Linear or microtile layout; the exp variant is not yet perf-optimized. |
| Transpose / Permute | Folded view with memtile fallback | Permutation of the final two axes only. AIE1 rejects permutations that require relayout because it has no memory-tile fallback. |
| Split / Slice | Direct or per-slice memtile | No Split/Slice kernel. A slice must be an exact union of complete producer-port regions. Cross-port slices require an unimplemented relay/repacking path and are rejected on every generation. Graph-boundary slices and chained views are not supported. |
| Concat | Direct or per-input memtile | No Concat kernel. Each consumer port must belong entirely to one input slice. Ports spanning multiple sources require an unimplemented gather/repacking path and are rejected on every generation. Graph-input/output-backed Concat and chained views are not supported. |
| Fanout / branching | Direct or per-consumer memtile | Each consumer is planned independently. Staging-compatible AIE1 buffer multicast is supported; two-way fanout is Vitis/aiesim verified, while higher fanout remains subject to DMA and routing resources. |
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
- Port-aligned Split/Slice and Concat compile into ordinary point-to-point connections; they do not add a view kernel
  or an additional data transformation.
- Chained folded views are not currently supported.
- Per-tensor static quantization is the primary supported quantization contract. Per-channel activation quantization is
  not supported.

## Transport

- Internal transport is realized as either a direct AIE connection or, on devices that provide memory tiles, one
  memory-tile stage. AIE1 fails explicitly when an incompatible layout requires relay, gather, or relayout.
- Direct transport means the endpoint staging descriptors are compatible. Physical shared-buffer aliasing is a
  separate placement optimization; direct transport may instead use tile DMA between distinct buffers.
- Fanout creates independently planned transport legs for each consumer. Reusing the same producer port disables
  exclusive shared-buffer aliasing but remains eligible for ADF buffer multicast.
- Split/Slice and Concat may use direct connections when producer and consumer port regions align exactly. An aligned
  leg with incompatible staging may use a memory tile on AIE-ML/AIE-MLv2; cross-port slices and consumer ports spanning
  multiple Concat sources are rejected before transport selection on every generation.
- Graph boundaries may expose multiple ports for partitioned tensors.
- Multi-stage relay transport is not implemented. Topologies requiring an additional relay stage fail explicitly.
- Internal AIE-to-PL-to-AIE bridge points and complete Versal system-link generation are not yet implemented.
