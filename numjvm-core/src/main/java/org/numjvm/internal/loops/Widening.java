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
}
