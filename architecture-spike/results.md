# Spike results

Machine for all numbers below unless stated otherwise: cloud VM, 2 vCPU Intel Xeon
@ 2.10 GHz, 7 GiB RAM, Linux 6.x, OpenJDK 25.0.4 (Ubuntu build), G1 GC, `-Xmx2g`.
A shared 2-vCPU VM is noisy; differences below roughly 20 % are not meaningful.

## S1 — Heap vs native allocation

Source: `src/spike/MemorySpike.java`.

### S1.1 Alignment of heap segments

| Segment | Layout | Result |
| --- | --- | --- |
| `byte[]`-backed | `JAVA_DOUBLE` (8-byte aligned) | **fails**: `IllegalArgumentException: Target offset 0 is incompatible with alignment constraint 8` |
| `byte[]`-backed | `JAVA_DOUBLE_UNALIGNED` | ok |
| `long[]`-backed | `JAVA_DOUBLE` | ok |
| `long[]`-backed | `JAVA_DOUBLE` at byte offset 4 | fails (expected) |
| `long[]`-backed | `maxByteAlignment()` | 8 |
| native, `allocate(n, 64)` | `maxByteAlignment()` | 128 (allocator returned more than requested) |

Heap buffers must therefore be backed by `long[]`; that gives 8-byte alignment, enough
for every dtype including the 8-byte components of `complex128`.

### S1.2 Allocation + release cost (manual harness, median of 7 rounds)

`heap` = `MemorySegment.ofArray(new long[n])`; `confined`/`shared` = arena created,
segment allocated with 64-byte alignment, arena closed; `auto` = `Arena.ofAuto()`,
released by GC. Heap B/op is the Java heap allocated per operation (native memory
is not visible here).

| float64 elements | heap | confined | shared | auto |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 83 ns | 87 ns | 38.6 µs | 418 ns |
| 128 | 225 ns | 177 ns | 34.8 µs | 1.0 µs |
| 1,024 | 1.4 µs | 280 ns | 33.5 µs | 5.6 µs |
| 65,536 | 53 µs | 10.3 µs | 41.7 µs | 217 µs |
| 1,048,576 | 6.3 ms | 322 µs | 358 µs | 1.3 ms |
| 16,777,216 | 13.7 ms | 56.3 ms | 53.4 ms | 66.9 ms |

Heap bytes per operation: heap = array size + ~90 B; confined/shared/auto = 144–168 B
independent of size.

### S1.3 Sequential read, aligned vs unaligned layout

1,048,576 float64 values, constant layouts, ns per element:

| Memory | `JAVA_DOUBLE` | `JAVA_DOUBLE_UNALIGNED` |
| --- | ---: | ---: |
| heap (`long[]`) | 0.63 | 0.65 |
| native | 0.67 | 0.66 |

No measurable difference. A first version passed the layout as a method parameter and
measured 6–9 ns/element: the JIT only intrinsifies a `VarHandle` access when the
layout is a constant. Kernels must therefore use `static final` layouts.

### S1.4 Heap vs automatic native crossover

| bytes | heap | auto | confined |
| ---: | ---: | ---: | ---: |
| 8 KiB | 1.5 µs | 4.8 µs | 1.5 µs |
| 16 KiB | 1.9 µs | 4.4 µs | 0.2 µs |
| 32 KiB | 17.4 µs\* | 4.9 µs | 0.4 µs |
| 64 KiB | 7.3 µs | 8.5 µs | 1.5 µs |
| 256 KiB | 26.5 µs | 32.2 µs | 5.6 µs |
| 1 MiB | 126 µs | 141 µs | 37 µs |
| 2 MiB | 228 µs | 1.1 ms | 75 µs |
| 8 MiB | 889 µs | 4.4 ms | 321 µs |

\* outlier, a GC cycle fell into the measured rounds.

### S1 observations

1. **Shared arenas have a fixed ~35 µs close cost** (thread handshake), independent of
   size. They must not be the default scope mode.
2. **Confined arenas are the cheapest way to get memory at every size above ~1 KiB**,
   because release is deterministic and nothing is left for GC.
3. **Automatic arenas are never cheaper than heap below 1 MiB** in this harness and
   2–5× more expensive above it, because release goes through `Cleaner` and GC.
   S1.2 shows the opposite at 8 MiB (heap 6.3 ms): there, a heap that is too small
   for the live set turned every allocation into a GC cycle. The manual harness
   cannot separate those effects; S5 repeats this with JMH.
4. Both answers to "heap or native for automatic arrays?" are therefore driven by
   GC behaviour, not by allocation cost. S4 measures the GC side.

## S2 — Minimal float64 array and odometer iterator

Source: `SpikeArray.java`, `Odometer.java`, `SpikeArrayCheck.java`. All 28 value
checks pass on heap and native memory: C-contiguous reads, `get(i, j)`, transpose
(shape `[4, 3]`, strides `[8, 32]`), positive and negative steps, broadcast with zero
strides (`[0, 8]`), 1-D → 3-D broadcast, 0-D and zero-length arrays.

| Access | Time | Heap bytes |
| --- | ---: | ---: |
| `get(i, j)` (rank-specialised) | 12.4 ns/op | 0 B/op |
| `getVarargs(i, j)` | 13.2 ns/op | 0 B/op |
| odometer sum, 256×256 transposed | 94.6 µs/op (1.44 ns/element) | 120 B/op |

Times include the harness lambda and a volatile write; only the relative numbers
matter.

### S2 observations

1. **The varargs accessor did not allocate here**: after inlining, escape analysis
   removed the index array. That is not a guarantee. It fails as soon as the call
   site is not inlined (large callers, megamorphic sites, interpreter, C1). The
   rank-specialised accessors are allocation-free by construction, so they stay the
   primary API and the `long[]` form is the general fallback (ADR-002).
2. **The odometer allocates once per iteration, not per element**: 120 B per traversal
   for its counter and offset arrays. That is acceptable for M0 and is what ADR-003
   adopts.
3. Byte strides make transpose, reverse and broadcast pure metadata changes; no case
   needed element-size arithmetic at access time.

## S3 — Strided add with broadcasting, pairwise sum

Source: `StridedAdd.java`, `SumReduction.java`, `KernelSpike.java`.

Correctness: all 18 checks pass — `(3,4) + (4,)`, transposed + contiguous,
reversed rows + column vector, pairwise sum for n ∈ {0, 1, 7, 8, 9, 127, 128, 129,
1000, 65 536, 1 048 577} against the closed form n(n−1)/2, sum over transposed and
broadcast views.

Accuracy, 2^24 copies of 0.1:

| Algorithm | Relative error |
| --- | ---: |
| pairwise (NumPy algorithm) | 1.4 × 10⁻¹⁶ |
| naive single accumulator | 2.5 × 10⁻¹⁰ |

Speed, ns per element (manual harness; `add` includes allocating and zeroing the
output, so it measures what a user call costs):

| elements | add fast path | add generic loop | add, one operand transposed | sum pairwise | sum naive |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.92 | 2.43 | 2.53 | 2.60 | 0.65 |
| 65,536 | 5.59 | 7.82 | 10.27 | 1.52 | 0.75 |
| 1,048,576 | 5.08 | 7.52 | 19.31 | 1.54 | 0.67 |
| 16,777,216 | 6.08 | 11.91 | 46.24 | 2.22 | 1.62 |

### S3 observations

1. **Contiguous fast path vs generic odometer loop: 1.3–2.0× faster** on identical
   contiguous inputs. Worth keeping, but the generic loop is not a cliff; M0 can ship
   the generic loop plus one contiguous fast path, and M1 adds dimension coalescing.
2. **Transposed operands degrade with size** (19–46 ns/element at ≥1M elements)
   because the inner loop strides through memory by a full row. This is the case
   loop reordering in M1 must fix; it is out of scope for M0.
3. **Pairwise sum is 2–4× slower than the naive loop** in this spike, mostly the
   per-block stride multiplications and the recursion, but it is six orders of
   magnitude more accurate and matches NumPy's results. M0 keeps pairwise (needed for
   oracle comparison); speeding it up is M1/M2 work (unit-stride specialisation,
   Vector API).
