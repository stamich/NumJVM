package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.DType;

/**
 * Exact comparisons between int64 and uint64 ({@code NUMPY_COMPATIBILITY}: NumPy 2's
 * {@code qQ->?} loops). Promoting the pair to float64, as arithmetic does, would round
 * values above 2<sup>53</sup> and make, for example, {@code 2^63 - 1 < 2^63} false.
 */
final class MixedComparisonLoops {

    private static final ValueLayout.OfByte I8 = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG_UNALIGNED;

    private MixedComparisonLoops() {
    }

    static void register(KernelRegistry.Registrar r) {
        r.binary("equal", DType.INT64, DType.UINT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, false, 0));
        r.binary("equal", DType.UINT64, DType.INT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, true, 0));
        r.binary("not_equal", DType.INT64, DType.UINT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, false, 1));
        r.binary("not_equal", DType.UINT64, DType.INT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, true, 1));
        r.binary("less", DType.INT64, DType.UINT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, false, 2));
        r.binary("less", DType.UINT64, DType.INT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, true, 2));
        r.binary("less_equal", DType.INT64, DType.UINT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, false, 3));
        r.binary("less_equal", DType.UINT64, DType.INT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, true, 3));
        r.binary("greater", DType.INT64, DType.UINT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, false, 4));
        r.binary("greater", DType.UINT64, DType.INT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, true, 4));
        r.binary("greater_equal", DType.INT64, DType.UINT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, false, 5));
        r.binary("greater_equal", DType.UINT64, DType.INT64, DType.BOOL, (a, pa, sa, b, pb, sb, o, po, so, n) ->
                run(a, pa, sa, b, pb, sb, o, po, so, n, true, 5));
    }

    /**
     * Compares signed {@code s} with unsigned {@code u} exactly: -1, 0 or 1.
     */
    static int compare(long s, long u) {
        if (s < 0 || u < 0) {
            // s negative: s < u; u >= 2^63 (negative as a long): u > every int64
            return -1;
        }
        return Long.compare(s, u);
    }

    /**
     * @param unsignedFirst {@code true} if {@code a} is the uint64 operand
     * @param kind          0 ==, 1 !=, 2 <, 3 <=, 4 >, 5 >=
     */
    private static void run(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                            long po, long so, long n, boolean unsignedFirst, int kind) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            long x = a.get(I64, pa);
            long y = b.get(I64, pb);
            int c = unsignedFirst ? -compare(y, x) : compare(x, y);
            boolean r = switch (kind) {
                case 0 -> c == 0;
                case 1 -> c != 0;
                case 2 -> c < 0;
                case 3 -> c <= 0;
                case 4 -> c > 0;
                default -> c >= 0;
            };
            o.set(I8, po, r ? (byte) 1 : (byte) 0);
        }
    }
}
