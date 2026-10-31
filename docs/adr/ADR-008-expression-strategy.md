# ADR-008: Expression strategy and optional fused-vector provider

Status: proposed for 0.5.0-alpha

The experimental expression API evaluates typed graph nodes in eager order. A
context may select a bounded block executor, a generated scalar executor, an
optional generated-vector executor, or an internal eager-kernel schedule. Node
promotion and rounding are independent of this selection. Float32 results round
at each float32 node, including before a subsequent float64 node; multiplication
and addition are not silently replaced by FMA or reassociated.

An invocation validates shapes, memory accessibility, dtype, output writability
and aliasing before writing. An identical elementwise input/output view may be
updated directly only when every overlapping input is safe for that traversal.
Other hazardous overlaps use independent storage. A cached plan must not retain
invocation arrays, scopes, owners, or raw addresses.

BLOCK keeps bounded scratch apart from a required alias-protection result.
Automatic selection must honor its temporary-memory budget; a whole-array eager
schedule may require full-sized intermediates. Explicit REQUIRED retains its
existing meaning: require eligible generated execution, and do not silently fall
back. Scalar generated execution satisfies REQUIRED; the optional vector provider
must not make JIT depend on the incubating Vector API.

Provider placement, byte limits, scalar/vector thresholds and exact diagnostics
remain implementation decisions. Measure compile and warm costs independently and
report a fallback reason without per-element logging. Core and JIT must load on
classpath and module path without the provider. Verify numerical parity and
memory-lifetime behavior for every supported strategy before selection.
