# ADR-009: Real dot and vecdot contractions

Status: proposed for 0.5.0-alpha

The optional linalg module will provide float32 and float64 dot and vecdot with a
portable Java implementation. Mixed float32/float64 inputs produce float64;
integer and complex operands are unsupported for this increment. Complex
conjugation is therefore a no-op in the supported vecdot surface.

Dot with a scalar multiplies elementwise; otherwise contract the left final axis
with the right final axis for a vector, or the right penultimate axis for higher
rank. Vecdot normalizes its selected axis for each input, contracts equal-length
axes and broadcasts the remaining dimensions. Zero-length contractions yield
zeros. Validate shape and axis before writing. Reference fixtures live in
`numjvm-tests-oracle/m3_1/contractions.json` and can be checked with
`numjvm-tests-oracle/check_m3_1_contract.py` before the APIs exist.

Allocate results in the first operand's open scope when present, following the
existing linalg ownership rules. Explicit output must have exact shape and dtype,
be accessible, writable and have nonoverlapping elements. Unlike an elementwise
operation, contraction can depend on values needed for later outputs: any
hazardous input/output overlap, including an identical view, requires independent
storage before copyback. Unexpected execution errors can leave a directly written
output partial; validation errors must precede writes.

Java, SIMD and native reduction orders may differ. For finite values, the initial
oracle check uses an absolute error bound `8 * gamma(k) * sum(abs(a*b)) + floor`,
where `gamma(k) = k*epsilon/(1-k*epsilon)` for `k*epsilon < 1`, and floor is
`8 * smallest_normal(dtype)`. For larger `k`, a separate high-precision test is
required instead of this bound. This proposal uses the result dtype epsilon and
does not establish bitwise agreement between reduction backends. Reference and
product calculations for the bound must avoid overflow, using a higher-precision
or scaled method when needed. NaNs, infinities, overflow and signed zero require
separate tests rather than the finite bound.

SIMD and LP64 OpenBLAS are optional accelerators. A missing DOT symbol must not
break GEMM. Validate dimensions, strides and native segment lifetimes; pack or
fall back if the native operation cannot represent an input safely. Required-mode
failure behavior and supported platform/provider diagnostics must be documented.
