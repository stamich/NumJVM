package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.numjvm.dtype.DType;

/** Mixed input/accumulator kernels for modular 64-bit integer reductions. */
public final class Widening {
    private Widening() { }
    /** Reads a source integer as the corresponding 64-bit bit pattern. */
    public static long read(MemorySegment memory, long offset, DType type) {
        return switch (type) {
            case BOOL -> memory.get(ValueLayout.JAVA_BYTE, offset) == 0 ? 0 : 1;
            case INT8 -> memory.get(ValueLayout.JAVA_BYTE, offset);
            case UINT8 -> memory.get(ValueLayout.JAVA_BYTE, offset) & 255L;
            case INT16 -> memory.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
            case UINT16 -> memory.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset) & 65535L;
            case INT32 -> memory.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
            case UINT32 -> memory.get(ValueLayout.JAVA_INT_UNALIGNED, offset) & 0xffffffffL;
            case INT64, UINT64 -> memory.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
            default -> throw new IllegalArgumentException("not integer: " + type);
        };
    }
    /** @return a mixed loop with a 64-bit accumulator and source-typed second input */
    public static BinaryLoop loop(DType input, boolean product) {
        return (a, pa, sa, b, pb, sb, out, po, so, n) -> {
            if (sa == 0 && so == 0) {
                long value = a.get(ValueLayout.JAVA_LONG_UNALIGNED, pa);
                for (long i = 0; i < n; i++) {
                    long x = read(b, pb + i * sb, input);
                    value = product ? value * x : value + x;
                }
                out.set(ValueLayout.JAVA_LONG_UNALIGNED, po, value);
            } else {
                for (long i = 0; i < n; i++) {
                    long value = a.get(ValueLayout.JAVA_LONG_UNALIGNED, pa + i * sa);
                    long x = read(b, pb + i * sb, input);
                    out.set(ValueLayout.JAVA_LONG_UNALIGNED, po + i * so, product ? value * x : value + x);
                }
            }
        };
    }

    private static double asDouble(MemorySegment memory, long offset, DType type) {
        long value = read(memory, offset, type);
        return type == DType.UINT64 ? org.numjvm.dtype.Unsigned.toDouble(value) : (double) value;
    }

    /** @return source-to-double sum with exactly the scalar eight-accumulator tree */
    public static double convertedSum(MemorySegment s, long off, long stride, long n, DType type) {
        if (n < 8) {
            double res = 0.0;
            for (long i = 0; i < n; i++) {
                res += asDouble(s, off + i * stride, type);
            }
            return res;
        } else if (n <= 128) {
            double r0 = asDouble(s, off, type);
            double r1 = asDouble(s, off + stride, type);
            double r2 = asDouble(s, off + 2 * stride, type);
            double r3 = asDouble(s, off + 3 * stride, type);
            double r4 = asDouble(s, off + 4 * stride, type);
            double r5 = asDouble(s, off + 5 * stride, type);
            double r6 = asDouble(s, off + 6 * stride, type);
            double r7 = asDouble(s, off + 7 * stride, type);
            long limit = n - n % 8;
            long i;
            for (i = 8; i < limit; i += 8) {
                long p = off + i * stride;
                r0 += asDouble(s, p, type);
                r1 += asDouble(s, p + stride, type);
                r2 += asDouble(s, p + 2 * stride, type);
                r3 += asDouble(s, p + 3 * stride, type);
                r4 += asDouble(s, p + 4 * stride, type);
                r5 += asDouble(s, p + 5 * stride, type);
                r6 += asDouble(s, p + 6 * stride, type);
                r7 += asDouble(s, p + 7 * stride, type);
            }
            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) {
                res += asDouble(s, off + i * stride, type);
            }
            return res;
        }
        long n2 = n / 2;
        n2 -= n2 % 8;
        return convertedSum(s, off, stride, n2, type) + convertedSum(s, off + n2 * stride, stride, n - n2, type);
    }

    /** @return a mixed integer-input, float64-accumulator addition loop */
    public static BinaryLoop meanSum(DType input) {
        return (a, pa, sa, b, pb, sb, out, po, so, n) -> {
            if (sa == 0 && so == 0) {
                out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, po,
                        a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, pa) + convertedSum(b, pb, sb, n, input));
            } else {
                for (long i = 0; i < n; i++) {
                    double value = a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, pa + i * sa);
                    out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, po + i * so, value + asDouble(b, pb + i * sb, input));
                }
            }
        };
    }
}
