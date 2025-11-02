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

## S4 — Lifetime soak: 100,000 short-lived off-heap arrays

Source: `LifetimeSoak.java`. Eight arrays alive at any time, each written and read
once. `-Xmx512m`, each mode in its own JVM. "Tracked native" is a spike-side counter
incremented on allocation and decremented by a `Cleaner` action.

32 KiB arrays (3.1 GiB allocated in total):

| Mode | Time | Peak RSS | GCs |
| --- | ---: | ---: | ---: |
| heap (`long[]`) | 563 ms | 270 MiB | 26 |
| `Arena.ofAuto()` | 1,008 ms | 575 MiB | 6 |
| `ofAuto()` + 256 MiB soft limit → `System.gc()` | 943 ms | 342 MiB | 13 (13 forced) |
| `Arena.ofConfined()` per array, closed immediately | 229 ms | 67 MiB | 1 |

1 MiB arrays (98 GiB allocated in total):

| Mode | Time | Peak RSS | GCs |
| --- | ---: | ---: | ---: |
| heap (`long[]`) | 6.2 s | 186 MiB | 958 |
| `Arena.ofAuto()` | 17.0 s | 567 MiB | 200 |
| `ofAuto()` + 256 MiB soft limit | 20.7 s | 310 MiB | 403 (403 forced) |
| `ofAuto()`, `-XX:MaxDirectMemorySize=128m` | 21.1 s | 183 MiB | 838 |
| `Arena.ofConfined()` | 2.4 s | 69 MiB | 1 |

A single `Arena.ofAuto().allocate(600 MiB)` with `-Xmx512m` fails with
`OutOfMemoryError: Cannot reserve 629145648 bytes of direct buffer memory (allocated: 0,
limit: 536870912)`, and succeeds with `-XX:MaxDirectMemorySize=1g`.

### S4 observations

1. **The unbounded-growth scenario from the analysis does not happen.** Native
   segments are reserved against the JDK's direct-memory limit
   (`-XX:MaxDirectMemorySize`, default = max heap). On reaching it, the JDK runs
   `System.gc()` and retries before throwing. RSS stayed below ~575 MiB in every
   automatic run; the hypothesis behind analysis finding #4 was wrong.
2. **The real costs are throughput and headroom.** Automatic native memory took
   2.7× longer than heap at 1 MiB, and a process can hold heap *plus* the same amount
   again in native memory by default.
3. **Our own soft limit duplicates what the JDK already does**: at the same peak RSS it
   was no faster than lowering `MaxDirectMemorySize`. It adds nothing as a safety
   mechanism. What the JDK does not give is *visibility*: there is no cheap API for
   "how much of that is NumJVM's?"
4. **Confined scopes are 4–7× faster than any GC-managed option** and keep RSS flat;
   they are the pattern to recommend for allocation-heavy loops.

Decision input for ADR-001: keep live-native-bytes accounting for `NJ.memoryStats()`;
make the soft limit opt-in and off by default; document `MaxDirectMemorySize` as the
real bound.

## S5 — JMH and NumPy baseline

Source: `benchmarks/src/spike/SpikeBenchmarks.java`, `benchmarks/numpy_baseline.py`.
JMH 1.37, 1 fork, 3 × 1 s warm-up, 5 × 1 s measurement, average time, `-prof gc`,
single thread. NumPy 2.5.3 on the same VM, `OMP_NUM_THREADS=1`, best of 5 `timeit`
rounds. Error columns are JMH's 99.9 % confidence half-width; on this shared VM they
are large for anything that allocates megabytes.

### Allocation (µs/op, heap bytes/op)

| float64 elements | heap `long[]` | `ofAuto()` | `ofConfined()` + close | NumPy `zeros` |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.18 ± 0.22 (8,280 B) | 3.35 ± 1.54 | 0.12 ± 0.05 | 0.25 |
| 65,536 | 59.5 ± 3.8 (524 KB) | 101 ± 29 | 11.5 ± 0.9 | 11.8 |
| 1,048,576 | 891 ± 127 (8.4 MB) | 1,981 ± 1,256 | 345 ± 72 | 350 |

### Kernels (µs/op; `add` includes allocating the output)

| elements | op | Java fast path | Java generic loop | NumPy | Java / NumPy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | add contiguous | 1.87 ± 0.31 | 2.98 ± 0.73 | 0.65 | 2.9× |
| 1,024 | add, one transposed | 3.00 ± 0.70 | — | 1.73 | 1.7× |
| 1,024 | sum (pairwise) | 0.67 ± 0.32 | naive 0.69 | 2.76 | 0.24× |
| 65,536 | add contiguous | 129 ± 20 | 208 ± 69 | 45.9 | 2.8× |
| 65,536 | add, one transposed | 202 ± 79 | — | 136 | 1.5× |
| 65,536 | sum (pairwise) | 43.7 ± 24.7 | naive 48.9 | 23.5 | 1.9× |
| 1,048,576 | add contiguous | 5,132 ± 4,113 | 4,814 ± 2,483 | 1,245 | 4.1× |
| 1,048,576 | add, one transposed | 8,974 ± 2,584 | — | 5,717 | 1.6× |
| 1,048,576 | sum (pairwise) | 787 ± 208 | naive 752 | 460 | 1.7× |

Allocation profile: every kernel benchmark is allocation-free apart from the output
array (sum: < 6 B/op, i.e. noise).

### S5 observations

1. **The 1.5× NumPy target from the proposal is not met by scalar Java loops** for
   contiguous `add` (2.8–4.1×); NumPy's loops are SIMD. `sum` is at 1.7–1.9× and
   *faster* than NumPy below a few thousand elements, where Python call overhead
   dominates. This supports the plan: M0 ships correct scalar loops, M2 adds the
   Vector API path, and the target is re-measured there.
2. **Pairwise and naive sum cost the same under JMH.** The 2–4× gap in S3 was a
   harness artifact (OSR compilation of the lambda loop). Pairwise costs nothing.
3. **Automatic native allocation is 1.7–2.8× slower than heap at every size** and
   much noisier; confined arenas are as fast as NumPy's `zeros`. For GC-managed
   arrays the heap is the better default well beyond 32 KiB.
4. The add fast path is not reliably faster than the generic loop at 1M elements;
   allocation and page faults on the output dominate there.
