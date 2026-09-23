# Operator and Feature Support


## Operators

| Operator / feature | Current support | Notes and limitations |
| --- | --- | --- |
| Dense / GEMM | Generation-dependent integer and float formats | Read-only RTP weights; optional bias and fused ReLU; configurable cascade parallelism. Exact precision/microtile combinations are device-specific. AIE-ML supports BF16 but not FP8; FP8 is available on AIE-MLv2. |
| Dynamic MatMul | Generation-dependent integer and float formats | Rank-2 GEMM ABI with device-specific precision/microtile combinations. A rank-2 RHS may be broadcast across compacted independent LHS axes; batched RHS MatMul with non-broadcast leading axes is rejected. FP8 is available on AIE-MLv2. |
| Elementwise Add | Generation-dependent integer and float formats | Exact-shape inputs for residual and elementwise connections; broadcasting is not supported. BF16 is supported on AIE-ML and FP8 on AIE-MLv2. |
| LayerNorm | Signed int8 | Last-axis normalization using integer mean/variance and reciprocal-square-root approximation. Input and output storage are currently signed int8 and require supported static quantization. |
| Softmax | Int8 to uint8/int16 | Accurate integer exponential or an opt-in surrogate (needs explicit parameters + QAT). Linear or microtile layout; the exp variant is not yet perf-optimized. |
| Conv2D | Signed int8, stride 1 | NHWC frames; any kernel that fits tile memory (validated to 7x7), asymmetric zero padding, `groups` (including depthwise), optional bias and fused ReLU. A layer spans several tiles through the Dense parallelism directive on the channel-block axis: `cas_length` splits the reduction along a cascade chain, `cas_num` splits the output channels ('inner') or the output rows ('outer') across chains. Row bands overlap by the window span, so a banded input comes from the graph boundary (the host delivers each band's window); a halo-free consumer such as a 1x1 conv inherits the bands, and nothing gathers them back, so a banded chain ends at the boundary. Batch > 1, stride, dilation and float are rejected explicitly. A flattened conv output feeds Dense directly (one chain only). A conv frame crossing the graph boundary must fit one 8-channel block per port. |
| Flatten / Reshape | Folded into the producing op | One sample to `[1, K]`, written directly by a producer that lists `flatten_2d` among its output views (today Conv2D), so no kernel or copy is instantiated. A view that ravels the axes in a different order hands that row order to the consuming family, which folds it into its constants. |
| Transpose / Permute | Folded view with memtile fallback | Permutation of the final two axes only. AIE1 rejects permutations that require relayout because it has no memory-tile fallback. |
| Split / Slice | Direct or per-slice memtile | No Split/Slice kernel. A slice must be an exact union of complete producer-port regions. Cross-port slices require an unimplemented relay/repacking path and are rejected on every generation. Graph-boundary slices and chained views are not supported. |
| Concat | Direct or per-input memtile | No Concat kernel. Each consumer port must belong entirely to one input slice. Ports spanning multiple sources require an unimplemented gather/repacking path and are rejected on every generation. Graph-input/output-backed Concat and chained views are not supported. |
| Fanout / branching | Direct or per-consumer memtile | Each consumer is planned independently. Staging-compatible AIE1 buffer multicast is supported; two-way fanout is Vitis/aiesim verified, while higher fanout remains subject to DMA and routing resources. |
| Constant scale | Quantized integer | Constant power-of-two output scaling is folded into Dense/MatMul output shifts. Arbitrary constant scaling is not yet supported. |
| Activation | ReLU | Fused into Dense and Conv2D; no standalone activation kernel. |

## Frontends

| Frontend | Current support | Notes and limitations |
| --- | --- | --- |
| ONNX | Recommended operator-level frontend | Supports explicit graphs composed from supported operators and quantized Q/DQ boundaries. A Transpose of an activation is a change of view, not of data: it composes into how the ONNX value sees its canonical tensor, and a consumer that needs the canonical order materializes the view as a folded transpose. A convolution therefore needs the NCHW view to be explicit -- export channels-last, as `input [N,H,W,C] -> Transpose(0,3,1,2) -> Conv` -- because a graph whose input is itself NCHW is not silently re-interpreted as NHWC. |
| hls4ml | Optional Keras/QKeras frontend | Dense stacks, Conv2D and DepthwiseConv2D: hls4ml is already channels-last, so it reaches the same conv2d contract as ONNX with no backend change (a depthwise layer's per-channel filters are rearranged into the compact group form). A SeparableConv2D is rejected; split it into its depthwise and pointwise convolutions. Install `hls4ml` separately when using this path. |


## Tensor and View Contracts

- Spatial activations are canonically NHWC. A frontend states how its own value order views that tensor; a consumer
  that needs the canonical order refuses a permuted view rather than silently reading permuted data.
- A spatial frame is the logical image inside a padded buffer: `TensorView.origin` says where the image starts, and the
  space around it is the zero border its consumer's `pads` asks for. Producer and consumer derive the frame from the
  tensor alone (its consumers' `pads` and `kernel_shape`), so neither needs to know the other's op type.
- Conv weights stay in the compact `[kh, kw, Cin/groups, Cout]` form in the IR; a variant expands the groups when it
  packs for its kernel.
- A windowed family declares its `SpatialAccess2D` (kernel, pads, strides, dilations); frame sizing, fusion and view
  folding ask the family what it supports instead of naming op types.
- A staging descriptor's `logical_origin` is the signed coordinate where that port's window starts in the tensor. The
  op that partitions the tensor publishes it, because only it knows how its ports divide the work; transport carries it
  and never recomputes it. Negative means the window opens on padding, and the boundary clips it against the tensor and
  leaves the rest zero -- which is what lets a row band start before the first image row.
- A boundary transfer that moves its whole buffer in the buffer's own order carries no access pattern: constraining it
  would force buffer descriptors that a plain linear transfer does not need (measured: a 2,880-byte frame needs 720 BD
  steps against a limit of 255).
- One padded frame serves one window. A tensor whose consumers read different windows, or a windowed consumer beside
  one that reads the image itself, needs a per-consumer view that transport does not materialize, and is refused
  before lowering.
- `TensorView` and every staging descriptor are verified where they are built: rank agreement, valid permutations,
  positive extents, offsets inside their buffer and a known storage encoding.
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
- A staging descriptor's `storage_layout` is how the buffer is encoded, not how it may be routed: `linear`,
  `microtiled` (tiles of the outer/inner plane, contiguous, walked as a grid) or `inner_blocked` (chunks of the inner
  axis with the chunk index outermost, as Conv frames use). Whether a memory tile can re-stage a layout is a separate
  question: an `inner_blocked` leg is routed point-to-point, and an explicit `io_route=memtile` on it fails saying so.
- A variant's `PortMap` is its port contract: per tensor the ADF group, port count, port kind (`buffer` or
  `stream`) and the kernel endpoints behind each hierarchical port, which is where DMA access constraints bind.
- `ports: stream` selects a variant whose data ports are core streams (Dense in both contracts and every
  cascade shape, and a single-tile Conv2D; other ops fail explicitly). A streamed conv carries the *logical* tensor on the
  wire -- rows, then columns, then channels, with no border, no padded channels and no computed-width tail -- and the
  kernel builds everything its compute core needs around that data, keeping one band of the image (its output rows plus
  the window span) rather than the whole of it. So a stream carries a wire order, while `inner_blocked` describes buffer
  memory. It is also the only way more than one channel block crosses the graph boundary in one port. Partitioning a
  streamed Conv2D across tiles is not implemented. It buys legality, not speed: see the measured
  comparison below. What a stream port carries depends on the op -- a Dense port carries its padded per-port tile in
  linear row order and the kernel re-tiles it in registers, while a Conv2D port carries the logical tensor described
  above and the kernel builds the padded frame itself. Either way there is no buffer to place, no bank contract, no DMA
  descriptor and no microtile on the wire. Stream legs
  are always direct: stream-to-stream requires identical staging descriptors, a PLIO feeds the padded tile (the host
  pads and trims), and a stream-to-buffer leg, a memory-tile route or a transposed view is rejected explicitly. The
  stream groups of one kernel must fit the core's stream ports (two in/out on AIE, one on AIE-ML). A microtile row
  must be a multiple of 16 bytes, or 8 bytes with an even M.
- Multi-stage relay transport is not implemented. Topologies requiring an additional relay stage fail explicitly.
- Internal AIE-to-PL-to-AIE bridge points and complete Versal system-link generation are not yet implemented.
