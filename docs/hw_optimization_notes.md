
This document records AIE-specific performance findings that are useful when tuning
aie4ml kernels. Architectural capabilities are kept separate from observations about a
particular generated binary.

# AIE1 optimization notes

## Accumulator and scheduling observations

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
potentially be improved by (TODO):

1. Reducing simultaneously live accumulators, including trying less loop unrolling.
2. Keeping values in `acc48` until the final shift, rounding, saturation, and `to_vector`.
3. Processing fewer output vectors together when accumulator pressure causes spills.
4. Restructuring loops to expose independent MACs alongside loads and conversions.
5. Avoiding temporary vectors or local arrays that unnecessarily extend live ranges.
6. Comparing equivalent `mac`, `mul`, and `mmul` formulations because they can produce
   materially different AIE1 schedules.

Performance conclusions should use both generated assembly and measured kernel cycles. PLIO
width must not be used to explain this gap: aie4ml selects a 128-bit PLIO for AIE1, AIE-ML,
and AIE-MLv2.
