package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.DType;

/**
 * Hand-written float64 inner loops: the reference shape that the generated loops follow.
 *
 * <p>Every loop has three paths, chosen per run from the strides: all operands contiguous
 * (indexed loop), one input broadcast along the run with the rest contiguous (the scalar
 * is read once), and the general strided loop. Layouts are {@code static final} constants
 * so the JIT can intrinsify every access.
 */
public final class Float64Loops {

    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;
    private static final long SIZE = 8;

    private Float64Loops() {
    }

    /**
     * Registers this class's loops.
     *
     * @param r the registry being initialised
     */
    static void register(KernelRegistry.Registrar r) {
        r.binary("add", DType.FLOAT64, DType.FLOAT64, DType.FLOAT64, Float64Loops::add);
        r.unary("copy", DType.FLOAT64, DType.FLOAT64, Float64Loops::copy);
    }

    /**
     * {@code out = a + b}.
     *
     * @param a   first input memory
     * @param pa  byte offset of the first input's first element
     * @param sa  first input byte stride
     * @param b   second input memory
     * @param pb  byte offset of the second input's first element
     * @param sb  second input byte stride
     * @param o   output memory
     * @param po  byte offset of the output's first element
     * @param so  output byte stride
     * @param n   number of elements
     */
    public static void add(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                           long po, long so, long n) {
        if (so == SIZE && sa == SIZE && sb == SIZE) {
            for (long i = 0; i < n; i++) {
                long k = i * SIZE;
                o.set(F64, po + k, a.get(F64, pa + k) + b.get(F64, pb + k));
            }
        } else if (so == SIZE && sa == 0 && sb == SIZE) {
            double x = a.get(F64, pa);
            for (long i = 0; i < n; i++) {
                long k = i * SIZE;
                o.set(F64, po + k, x + b.get(F64, pb + k));
            }
        } else if (so == SIZE && sa == SIZE && sb == 0) {
            double y = b.get(F64, pb);
            for (long i = 0; i < n; i++) {
                long k = i * SIZE;
                o.set(F64, po + k, a.get(F64, pa + k) + y);
            }
        } else {
            for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
                o.set(F64, po, a.get(F64, pa) + b.get(F64, pb));
            }
        }
    }

    /**
     * {@code out = in}.
     *
     * @param in  input memory
     * @param pi  byte offset of the input's first element
     * @param si  input byte stride
     * @param o   output memory
     * @param po  byte offset of the output's first element
     * @param so  output byte stride
     * @param n   number of elements
     */
    public static void copy(MemorySegment in, long pi, long si, MemorySegment o, long po, long so, long n) {
        if (so == SIZE && si == SIZE) {
            MemorySegment.copy(in, pi, o, po, n * SIZE);
        } else if (so == SIZE && si == 0) {
            double x = in.get(F64, pi);
            for (long i = 0; i < n; i++) {
                o.set(F64, po + i * SIZE, x);
            }
        } else {
            for (long i = 0; i < n; i++, pi += si, po += so) {
                o.set(F64, po, in.get(F64, pi));
            }
        }
    }
}
