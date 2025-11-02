# ADR-001: Allocation model and heap/native threshold

- Status: Accepted
- Date: 2025-11-02
- Evidence: [`architecture-spike/results.md`](../../architecture-spike/results.md), sections S1, S4, S5

## Context

NumJVM stores array data in `java.lang.foreign.MemorySegment`s. A segment can be
backed by a Java array on the heap or by native memory from an `Arena`. The proposal
suggested a hybrid allocator with a 32 KiB heap threshold, scoped arenas for
deterministic release, and raised two concerns: heap-segment alignment, and native
memory from automatic arenas growing because the GC does not see it.

The spike measured:

1. A `byte[]`-backed segment cannot be read with aligned `double` layouts; `long[]`
   can. Heap segments report `maxByteAlignment() == 8` (S1.1).
2. Automatic native allocation is 1.7–2.8× slower than heap allocation at 8 KiB,
   512 KiB and 8 MiB, with much higher variance. Confined arenas are 5–10× faster
   than heap and on par with NumPy's `zeros` (S5).
3. Shared arenas pay a fixed ~35 µs on close for the thread handshake (S1.2).
4. Native segments are reserved against `-XX:MaxDirectMemorySize` (default: max
   heap). Reaching it triggers `System.gc()` before an `OutOfMemoryError`. In the
   100k-array soak, RSS stayed bounded in every mode (S4). A NumJVM-side soft limit
   gave no benefit over lowering `MaxDirectMemorySize`.

## Decision

1. **Heap buffers are `long[]`-backed** (`ceil(bytes / 8)` longs, sliced to the
   requested size). That gives 8-byte alignment for every dtype, including complex128
   components. The largest heap buffer is therefore about 16 GiB.
2. **Native buffers are allocated with 64-byte alignment**, ready for SIMD (M2) and
   for native libraries (BLAS, Arrow, DLPack).
3. **Arrays not created in a scope use the `HybridAllocator`**: requests up to
   the threshold go to the heap, larger ones to an automatic arena.
   - **Default threshold: 1 MiB**, not the proposal's 32 KiB. Below 1 MiB native
     allocation has no measured advantage. Above it, native memory avoids copying
     large arrays during GC and can be handed to native code without a copy from M2 on.
   - Configurable with `-Dnumjvm.memory.heapThreshold=<bytes>`; `0` means "always
     native", a negative value is rejected.
4. **Arrays created in an `NJScope` are always native**, allocated from the scope's
   arena regardless of size. Release is deterministic, and liveness is enforced by
   FFM itself, so element access needs no extra "is the scope open?" check.
   - `NJ.scope()` is **confined** (single thread, cheapest close).
   - `NJ.scopeShared()` is opt-in for multi-threaded use and documents the ~35 µs
     close cost.
5. **`NJScope.keep(array)` copies** the array into hybrid-allocated automatic storage
   (C order). FFM cannot move a segment between arenas, so zero-copy transfer is not
   offered.
6. **NumJVM counts the native bytes it allocates** (live, peak, total count) and
   exposes them through `NJ.memoryStats()`. The JDK bounds native memory but does not
   say how much of it is NumJVM's.
7. **The soft limit is opt-in** (`-Dnumjvm.memory.nativeSoftLimit=<bytes>`, unset by
   default). When set and exceeded, automatic allocations above the heap threshold
   fall back to the heap while they fit in a `long[]`. The documented bound on native
   memory is `-XX:MaxDirectMemorySize`.

## Consequences

- Positive: no alignment failure modes; scoped code gets NumPy-level allocation
  cost; memory bounds come from a JDK mechanism users already know.
- Positive: most small and medium arrays live on the heap and behave like ordinary
  Java objects (no native accounting, no `Cleaner` work).
- Negative: arrays between 32 KiB and 1 MiB are copied by the GC if they survive. If
  M1 benchmarks show that this matters, the threshold is a single property.
- Negative: scoped arrays are not readable from other threads unless the scope is
  shared; this must be prominent in the documentation.
- Follow-up: re-measure the threshold in M2 once SIMD kernels make memory bandwidth,
  not allocation, the bottleneck.
