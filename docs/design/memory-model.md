# Memory model

NumJVM arrays live in `java.lang.foreign.MemorySegment`s. This document describes who
owns that memory, how long it stays valid, and what happens when it is accessed too
late or from the wrong thread. The decisions behind it are in
[ADR-001](../adr/ADR-001-allocation-model.md).

## Ownership

```text
MemoryOwner (automatic | NJScope)
      │ decides lifetime
   Buffer  ── MemorySegment (heap long[] or native arena memory)
      │ strong reference
 ┌────┼─────────┐
NDArray  view   view of view ...
```

- Every **buffer** has exactly one **owner** and one segment. Buffers are internal
  (`org.numjvm.internal.memory`); the public API never hands out a writable segment,
  because that would bypass the read-only flag of views.
- Every **array and view** holds a strong reference to its buffer. A view can therefore
  never outlive the memory it reads, and creating views never shortens a lifetime.
- There are two owners:
  - **automatic** memory stays valid while any array that uses it is reachable and is
    reclaimed by the garbage collector afterwards;
  - an **`NJScope`** frees all its memory when it is closed.

## Where memory comes from

| Created by | Size | Memory | Released |
| --- | --- | --- | --- |
| `NJ.*` factories, `copy()`, `astype()`, `Npy.read(path)`, kernels | ≤ heap threshold (1 MiB) | Java heap, `long[]`-backed | by the GC |
| same | > heap threshold | native, automatic arena, 64-byte aligned | by the GC (via the arena's cleaner) |
| `scope.*` factories, `Npy.read(path, scope)`, `ValidationKernels.add(a, b, scope)` | any | native, the scope's arena, 64-byte aligned | when the scope closes |
| `Npy.map(path, scope)` | any | memory-mapped file, read-only | when the scope closes |

Heap buffers are backed by `long[]` rather than `byte[]`: heap segments are only as
aligned as their array elements, and a `byte[]` segment cannot be read with aligned
`double` layouts. Element access uses unaligned layouts anyway (ADR-002); alignment
matters for SIMD and native libraries later.

### Settings

| System property | Default | Meaning |
| --- | --- | --- |
| `numjvm.memory.heapThreshold` | `1048576` | Largest automatic allocation, in bytes, placed on the heap. `0` sends every non-empty array off-heap. |
| `numjvm.memory.nativeSoftLimit` | unset | If set, automatic allocations that would push NumJVM's live native bytes past it go to the heap instead (while they fit in a `long[]`). |

The hard limit on native memory is the JVM's `-XX:MaxDirectMemorySize` (default: the
maximum heap size). Native arena memory is reserved against it; when it is reached the
JDK runs a GC and retries before failing, and NumJVM reports the failure as a
`MemoryException`. `NJ.memoryStats()` shows how much of it NumJVM holds:

```java
MemoryStats s = NJ.memoryStats();
s.liveNativeBytes();   // allocated and not yet released
s.peakNativeBytes();
s.nativeAllocations();
```

## Scopes

```java
NDArray result;
NDArray view;
try (NJScope scope = NJ.scope()) {
    NDArray a = scope.zeros(DType.FLOAT64, 1000, 1000);
    view = a.transpose();                 // a view: same memory, same lifetime
    result = scope.keep(view);            // a copy into automatic memory
}
result.getDouble(0, 0);                   // fine
view.getDouble(0, 0);                     // MemoryException: confined NJScope#1 (thread main) is closed ...
```

- `close()` frees everything at once and is idempotent.
- After closing, **every** array and view created from the scope throws
  `MemoryException` on access. `isAlive()` reports the state without throwing, and
  `toString()` prints `array(<inaccessible: ...>, shape=..., dtype=...)`.
- `keep(array)` always copies. FFM cannot move a segment from one arena to another, and
  a zero-copy promise that depends on implementation details would be worse than an
  honest copy.
- Allocation in a scope is cheap: in the spike (S5), a confined arena allocated and
  freed an 8 MiB array (1,048,576 float64 values) 2.6× faster than the heap and 5.7×
  faster than an automatic arena. Prefer scopes for temporaries in loops.

## Threads

| Memory | Readable from | Closable from |
| --- | --- | --- |
| automatic (heap or native) | any thread | n/a |
| `NJ.scope()` / `NJ.scopeConfined()` | the creating thread only | the creating thread only |
| `NJ.scopeShared()` | any thread | any thread |

Access from a thread a confined scope does not allow throws `MemoryException`
("confined to another thread"), as does closing it from such a thread; the scope then
stays open. A shared scope's `close()` costs about 35 µs (a handshake with all threads),
independent of size.

Sharing makes access **possible**, not **safe**: NumJVM does not synchronise element
writes. Concurrent writes to the same elements, or a write concurrent with a read of the
same element, need external synchronisation, exactly as for Java arrays.

## Failure modes

| Situation | Exception |
| --- | --- |
| Access after the scope closed | `MemoryException` (cause: `IllegalStateException`) |
| Access to a confined scope's array from another thread | `MemoryException` (cause: `WrongThreadException`) |
| Allocation from a closed scope | `MemoryException` |
| Native memory exhausted (`MaxDirectMemorySize`) | `MemoryException` naming the flag |
| Size overflow when computing bytes | `ShapeException` |
| Creating a view on closed memory | `MemoryException` |
