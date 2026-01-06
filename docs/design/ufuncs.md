# Ufuncs

A ufunc ("universal function") applies one operation element by element, with broadcasting and type promotion. NumJVM
0.2 has 41, all constants of `org.numjvm.ufunc.Ufuncs`; `NJ` and `NDArray` have shortcuts for them
(`NJ.add(a, b)`, `a.add(b)`, `NJ.sqrt(x)`, `x.sqrt()`). Reductions (`sum`, `max`, ...) are built on the same
machinery. This document describes how a call runs; the promotion rules are in [Type promotion](promotion.md),
the traversal in [ADR-004](../adr/ADR-004-iteration-plans.md).

## The ufuncs

| Group | Ufuncs |
|-------|--------|
| Arithmetic | `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `power`, `negative`, `positive`, `absolute`, `conjugate`, `sign`, `square` |
| Extrema | `maximum`, `minimum` (NaN propagates), `fmax`, `fmin` (NaN ignored) |
| Comparison | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` |
| Logical | `logical_and`, `logical_or`, `logical_xor`, `logical_not` |
| Bitwise | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert` |
| Functions | `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh` |
| Rounding | `floor`, `ceil`, `trunc`, `rint`, plus `round(x, decimals)` in `Rounding` |

`Ufunc.types()` lists the loops of a ufunc in NumPy's notation (`"int8,int8->int8"`), and `Ufuncs.byName("add")` finds
one by its NumPy name.

## Anatomy of a call

`NJ.add(a, b)`, that is `Ufuncs.ADD.apply(a, b)`, runs these steps:

1. **Promote.** `TypePromotion.resultType(a.dtype(), b.dtype())` gives the common dtype; a Java `long`, `double` or
   `Complex` argument is a weak scalar and promotes by kind only.
2. **Resolve a loop.** The ufunc takes the registered loop whose input dtype equals the common dtype, else the first
   loop, in NumPy's loop order, that the common dtype casts to safely (ADR-005). A ufunc may override this:
   `divide` sends integers and bool to its float64 loop. The resolution is cached per common dtype. No loop is a
   `DTypeException` naming the ufunc and the operand dtypes, as NumPy's message does.
3. **Convert.** Operands whose dtype differs from the loop's are converted into temporaries, keeping their memory
   order.
4. **Allocate.** The result gets the broadcast shape, the loop's output dtype, and the memory order of the operands
   (NumPy's `order='K'`: transposed operands give a transposed, contiguous result). It lives in the scope of the first
   operand allocated in an open `NJScope`, otherwise in automatically managed memory (heap below 1 MiB, native above).
5. **Iterate.** `LoopExecutor` builds an `IterationPlan` over the operands (drop length-1 axes, order axes by stride,
   coalesce) and calls the loop once per run.

## Loops

A loop is an interface with one method that processes a run of `n` elements:

```java
void run(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment out, long po, long so, long n);
```

Each run is handled by the fastest of these paths, chosen from the strides:

| Path | Condition | Used for |
|------|-----------|----------|
| reduction | output aliased to the first input, both stride 0 | `ufunc.reduce` along a run; `add` sums it pairwise |
| contiguous | every stride equals the element size | C, F and transposed operands after coalescing |
| broadcast first / second | one input has stride 0, the rest contiguous | `a + 2.0`, a row against a matrix |
| contiguous output | the output is contiguous, an input is not | strided views |
| strided | anything else | |

Loops come from three places, all registered in `KernelRegistry` in NumPy's order (bool, int8, uint8, ..., float64,
bfloat16, complex64, complex128):

- **Generated loops** (584): `numjvm-core/src/codegen` holds a table of operations (`Ops`) with one Java
  expression per dtype family, and templates for each path. The Gradle task `generateKernels` writes them to
  `build/generated/sources/kernels`. Narrow integers compute in `int` and wrap on store; float16 and bfloat16 compute
  in `float`; uint64 uses `Long`'s unsigned methods.
- **Complex loops** (`ComplexArithmetic`, `ComplexMath`): hand-written, because each element has two parts.
- **Mixed comparisons** (`MixedComparisonLoops`): int64 against uint64, compared exactly.

## Semantics worth knowing

- **Integers wrap** on overflow, as in NumPy. Integer division and remainder by zero give 0 (NumPy gives 0 with a
  warning; NumJVM has no warnings). A negative integer exponent is a `ValueException`.
- **floor_divide and remainder** follow Python: the remainder has the sign of the divisor.
- **maximum/minimum** propagate NaN; on equal operands the float32 and float64 loops return the second, the float16
  loop the first (`maximum(0.0, -0.0)`), as NumPy's loops do.
- **Comparisons** of signed integers with uint64, or of integers with an out-of-range Java `long`, are exact: NumJVM
  does not round through float64.
- **Complex**: `multiply` and `square` use the textbook formula, `divide` Smith's algorithm, `absolute` `hypot` and
  returns a real dtype; `exp`, `log` and `sqrt` follow C99 Annex G for infinities and NaN; `sign` is `z / |z|`;
  `equal` compares both parts. NumPy's complex ordering (`less`, `maximum`, ...), `power` and trigonometric loops are
  not implemented yet: they are `DTypeException`s, tagged `NUMJVM_SPECIFIC` in the oracle.
- **Accuracy (decision D6)**: arithmetic, comparisons, `sqrt` and rounding are exact and match NumPy bit for bit.
  `power`, `exp`, `log`, `sin`, `cos` and `tanh` use `java.lang.Math` (complex ones are built on it) and may differ from
  NumPy by up to 4 units in the last place; complex `divide`, `absolute` and `sign` by up to 2. The measured worst case
  over the oracle is 2 ulp.

## Output arrays and in-place operations

Every ufunc accepts an output: `Ufuncs.ADD.apply(a, b, out)`, `Reductions.sum(x, axes, keepdims, out)`, and in place
`a.addInPlace(b)`. As in NumPy, `out` must have exactly the result shape (it is not broadcast), be writeable, and accept
the result dtype with `SAME_KIND` casting (an int8 array accepts an int16 sum but not a float64 one). When `out` has the
loop's dtype and does not overlap an input in an unsafe way, the loop writes into it directly; otherwise the result goes
through a temporary. Overlap is safe when `out` is one of the inputs viewed identically, which covers in-place
operations.

## Reductions

`BinaryUfunc.reduce(x, axes, keepdims)` (NumPy's `ufunc.reduce`) and `Reductions` (`sum`, `prod`, `max`, `min`,
`mean`, `any`, `all`) run the ufunc's loop on the reduction path: the output has stride 0 on the reduced axes and is
also the first input. The iteration plan lets the input decide the axis order, so a transposed or strided array is
read sequentially.

- A ufunc with an identity (`add` 0, `multiply` 1, `logical_and` true, `bitwise_and` all ones) starts from it, so an
  empty reduction gives the identity. `maximum` and `minimum` start from the first element; empty input is a
  `ValueException`.
- Floating-point and complex `add` reductions use NumPy's pairwise summation (blocks of 128 elements, eight
  accumulators), so `sum` matches NumPy bit for bit. float16 and bfloat16 accumulate in float32.
- `sum` and `prod` of bool and integers narrower than 64 bits accumulate in int64 (uint64 for unsigned).
- `mean` sums (in float64 for integers, float32 for float16) and divides by the count as an int64, as NumPy's intp:
  float32 means are divided in float64 and complex64 means in complex128, then rounded back.
- Several axes are reduced at once only for associative operations; others (`subtract`, `divide`) reduce one axis.

## Memory

Ufuncs allocate their result and, when operand dtypes differ from the loop's, temporaries. Inside a scope those live in
the scope; outside, in automatic memory. Loops themselves allocate nothing: the JMH benchmarks show only the result's
bytes per call under `-prof gc`.
