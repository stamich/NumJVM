# ADR-002: Element access API

- Status: Accepted
- Date: 2025-11-02
- Evidence: [`architecture-spike/results.md`](../../architecture-spike/results.md), sections S1.3 and S2

## Context

Users need to read and write single elements, and M1 kernels must not pay for abstraction per element. The proposal
sketched `get(long... idx)` / `set(...)`. The spike found:

1. Value layouts must be compile-time constants for the JIT to intrinsify segment access. With a layout passed as a
   parameter, access cost 10× more (S1.3).
2. Aligned and unaligned layouts cost the same on aligned memory (S1.3).
3. A varargs accessor did *not* allocate in the spike, because escape analysis removed the array after inlining. That
   only holds when the call site is inlined.
4. Rank-specialised accessors are allocation-free by construction (S2).

## Decision

1. **Typed, rank-specialised accessors are the primary API**:

   ```java
   double getDouble();                       // 0-D
   double getDouble(long i);
   double getDouble(long i, long j);
   double getDouble(long i, long j, long k);
   double getDouble(long[] index);           // any rank
   void   setDouble(long i, long j, double value);   // value always last
   ```

   The same set exists for `long` (`getLong`/`setLong`) and `boolean`
   (`getBoolean`/`setBoolean`). The general form takes a `long[]`, not varargs, so an allocation at the call site is
   visible in the source.
2. **Unaligned value layouts, held in `static final` fields**, are used for all element access. Buffers are still
   allocated aligned (ADR-001) for SIMD and native interop; access never depends on it.
3. **Bounds checking is split**:
    - Each index is checked against its axis length (one unsigned comparison per axis). Negative indices are rejected;
      Python-style negative indexing is available through `Slice`, not through accessors.
    - The array's whole byte extent is validated **once**, when the array or view is created. Accessors do no other
      range arithmetic; the segment's own bounds check remains as a last line of defence.
4. **Conversion rules** (documented on `NDArray`):

   | Accessor | Accepts | Rule |
      | --- | --- | --- |
   | `getDouble` | every real dtype | bool → 0/1; uint64 read as unsigned; float16/bfloat16 widened |
   | `getLong` | bool and integer dtypes | uint64 returns the raw 64 bits |
   | `getBoolean` | every real dtype | non-zero → `true` |
   | `setDouble` | every real dtype | integers: truncation toward zero (NumPy `astype` semantics); bool: `v != 0` |
   | `setLong` | every real dtype | narrowing for smaller integers, rounding to nearest for floats |
   | `setBoolean` | every real dtype | `true` → 1 |
   | any | complex64/128 | `DTypeException` (storage-only in M0) |

5. **Writes check the `WRITEABLE` flag** and throw `ReadOnlyArrayException`
   (broadcast views are read-only).
6. **FFM lifecycle exceptions are translated**: `IllegalStateException` (closed arena)
   and `WrongThreadException` become `MemoryException` with a message that names the scope.
7. The API is marked `@Experimental` for 0.1.0 so it can still change after M1 kernels have used it.

## Consequences

- Positive: zero allocation per access regardless of JIT decisions; one code path per accessor and dtype.
- Positive: hot kernels (M1) bypass these accessors and use the same constant layouts directly on the segment, with the
  same extent guarantees.
- Negative: four overloads per type and operation is verbose. Ranks above 3 use the
  `long[]` form.
- Negative: accessors dispatch on dtype per call (one `switch`). That is fine for scalar access, and kernels avoid it.
