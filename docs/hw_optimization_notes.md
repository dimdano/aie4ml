
This document records AIE-specific performance findings that are useful when tuning
aie4ml kernels. Architectural capabilities are kept separate from observations about a
particular generated binary.

# Hardware optimization notes

## DMA descriptor budgets

DMA descriptor use depends on both the tiling pattern and its repeat extents. A larger
contiguous transfer normally changes a BD count field, but a traversal with more nested
dimensions than the DMA can encode is expanded into multiple BDs. For example, reordering a
`256x32` linear shard into `2x8` microtiles produces 4 column blocks and 128 row groups; Vitis
expands this to `4 * 128 = 512` BDs.

BD budgets are local resources. On AIE1, the compiler enforces 16 S2MM descriptors in the
memory module serving a port; descriptors elsewhere in the array cannot satisfy that port.
When this limit appears:

1. Prefer a layout whose packed order already matches linear I/O, so no access constraint is
   emitted. A one-row microtile is one example.
2. Reduce the repeated extent (for example, a smaller per-invocation batch or outer dim)
3. Split work across additional ports or tiles only when that also distributes the local DMA
   resources.
4. Use a memory tile or explicit relayout kernel only after checking latency and throughput.

Do not estimate BD use from tensor bytes alone. Inspect the descriptor traversal, direction,
padding, ping-pong buffering, and the target generation's local DMA limits.

## AIE1 accumulator and scheduling observations

AIE1 integer MACs use native `acc48` or `acc80` accumulator lanes; there is no native
`acc32` path analogous to AIE-ML. Inputs and outputs are normally held in narrower vector
lanes, so a kernel must eventually widen values into accumulators and shift, round, and
saturate results back to vectors. For example, the current aie4ml's (AIE1) Softmax compilation
emits substantially more of these conversions than its AIE-ML `acc32` counterpart.

The generated AIE1 Softmax binaries also show accumulator/vector spills to local memory and
many unfilled VLIW issue slots. These are observations about the current kernel and Vitis
2025.2 schedule, not architectural requirements: AIE1 does not necessarily spill, and other
kernels may schedule differently. Spills are pure overhead and typically indicate excessive
live values. Empty VLIW slots commonly indicate dependency chains, operations restricted to
particular slots, or insufficient independent work around loads and conversions.

The architectural differences cannot be eliminated, but the generated schedule can
potentially be improved by:

1. Reducing simultaneously live accumulators, including trying less loop unrolling.
2. Keeping values in `acc48` until the final shift, rounding, saturation, and `to_vector`.
3. Processing fewer output vectors together when accumulator pressure causes spills.
4. Restructuring loops to expose independent MACs alongside loads and conversions.
5. Avoiding temporary vectors or local arrays that unnecessarily extend live ranges.
6. Comparing equivalent `mac`, `mul`, and `mmul` formulations because they can produce
   materially different AIE1 schedules.

Treat an unexpectedly large stack requirement as a possible spill symptom before raising the
configured stack size. In one AIE1 Dense schedule, keeping two independent `acc48` results live
made Vitis request a 4096-byte stack; processing one result at a time reduced usage to 64 bytes.
Prefer reducing live state when spill removal offsets any lost parallelism, and confirm the result
in `report_stack.txt`. A fixed global stack increase is inappropriate because stack pressure is
kernel- and schedule-dependent.

Performance conclusions should use both generated assembly and measured kernel cycles. PLIO
width must not be used to explain this gap: aie4ml selects a 128-bit PLIO for AIE1, AIE-ML,
and AIE-MLv2.

Using microtilings in dense/matmul other than the default ones often lowers performance. (for example AIE1 Dense default `2x8x8` is 3x faster than the `1x16x8` schedule)
