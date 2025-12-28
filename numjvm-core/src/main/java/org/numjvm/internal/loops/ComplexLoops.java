package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.DType;

/**
 * Copy and cast loops between complex64 and complex128, and from complex to bool (true when
 * either part is non-zero). Conversions from real dtypes to complex are generated with the
 * real loops; complex to any other real dtype has no loop, because it would discard the
 * imaginary part ({@code NUMJVM_SPECIFIC}: NumPy discards it with a warning). Arithmetic is
 * in {@link ComplexArithmetic}.
 */
final class ComplexLoops {

    private static final ValueLayout.OfByte I8 = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    private ComplexLoops() {
    }

    static void register(KernelRegistry.Registrar r) {
        r.unary("copy", DType.COMPLEX64, DType.COMPLEX64, ComplexLoops::copy64);
        r.unary("copy", DType.COMPLEX128, DType.COMPLEX128, ComplexLoops::copy128);
        r.unary("cast:complex64", DType.COMPLEX64, DType.COMPLEX64, ComplexLoops::copy64);
        r.unary("cast:complex128", DType.COMPLEX64, DType.COMPLEX128, ComplexLoops::widen);
        r.unary("cast:complex64", DType.COMPLEX128, DType.COMPLEX64, ComplexLoops::narrow);
        r.unary("cast:complex128", DType.COMPLEX128, DType.COMPLEX128, ComplexLoops::copy128);
        r.unary("cast:bool", DType.COMPLEX64, DType.BOOL, ComplexLoops::nonZero64);
        r.unary("cast:bool", DType.COMPLEX128, DType.BOOL, ComplexLoops::nonZero128);
        ComplexArithmetic.register(r);
    }

    static void nonZero64(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            boolean nz = a.get(F32, pa) != 0f || a.get(F32, pa + 4) != 0f;
            o.set(I8, po, nz ? (byte) 1 : (byte) 0);
        }
    }

    static void nonZero128(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            boolean nz = a.get(F64, pa) != 0.0 || a.get(F64, pa + 8) != 0.0;
            o.set(I8, po, nz ? (byte) 1 : (byte) 0);
        }
    }

    static void copy64(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        copy(a, pa, sa, o, po, so, n, 8);
    }

    static void copy128(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        copy(a, pa, sa, o, po, so, n, 16);
    }

    private static void copy(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n, int size) {
        if (sa == size && so == size) {
            MemorySegment.copy(a, pa, o, po, n * size);
        } else {
            for (long i = 0; i < n; i++, pa += sa, po += so) {
                MemorySegment.copy(a, pa, o, po, size);
            }
        }
    }

    static void widen(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            o.set(F64, po, a.get(F32, pa));
            o.set(F64, po + 8, a.get(F32, pa + 4));
        }
    }

    static void narrow(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            o.set(F32, po, (float) a.get(F64, pa));
            o.set(F32, po + 4, (float) a.get(F64, pa + 8));
        }
    }
}
