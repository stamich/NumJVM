package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.DType;

/**
 * Element-wise loops for complex64 and complex128: add, subtract, multiply, divide,
 * negative, positive, conjugate, square, absolute, equal and not_equal.
 *
 * <p>Each loop computes in the precision of its dtype ({@code float} parts for complex64),
 * with the formulas of NumPy's scalar loops ({@code loops.c.src}):
 * <ul>
 *   <li>multiply and square: {@code (ar·br − ai·bi) + (ar·bi + ai·br)j}, without fused
 *       multiply-add; NumPy's SIMD loops fuse on some CPUs, which changes the last bit of a
 *       part that cancels and can turn an overflow into ±inf instead of NaN</li>
 *   <li>divide: Smith's algorithm, scaling by the larger part of the divisor; a zero divisor
 *       divides each part by +0.0, giving inf or NaN parts as NumPy does</li>
 *   <li>absolute: {@code hypot(re, im)}, a real result that never overflows needlessly</li>
 *   <li>equal: both parts equal ({@code ==}, so NaN is unequal to itself)</li>
 * </ul>
 * Every loop reads an element's inputs before writing its output, so the reduction path
 * (output aliased to the first input, both with stride 0) accumulates in order; add sums
 * that path pairwise, part by part, as NumPy does.
 */
final class ComplexArithmetic {

    private static final ValueLayout.OfByte I8 = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    private static final DType C8 = DType.COMPLEX64;
    private static final DType C16 = DType.COMPLEX128;

    private ComplexArithmetic() {
    }

    static void register(KernelRegistry.Registrar r) {
        r.binary("add", C8, C8, C8, ComplexArithmetic::add64);
        r.binary("add", C16, C16, C16, ComplexArithmetic::add128);
        r.binary("subtract", C8, C8, C8, ComplexArithmetic::subtract64);
        r.binary("subtract", C16, C16, C16, ComplexArithmetic::subtract128);
        r.binary("multiply", C8, C8, C8, ComplexArithmetic::multiply64);
        r.binary("multiply", C16, C16, C16, ComplexArithmetic::multiply128);
        r.binary("divide", C8, C8, C8, ComplexArithmetic::divide64);
        r.binary("divide", C16, C16, C16, ComplexArithmetic::divide128);
        r.binary("equal", C8, C8, DType.BOOL, ComplexArithmetic::equal64);
        r.binary("equal", C16, C16, DType.BOOL, ComplexArithmetic::equal128);
        r.binary("not_equal", C8, C8, DType.BOOL, ComplexArithmetic::notEqual64);
        r.binary("not_equal", C16, C16, DType.BOOL, ComplexArithmetic::notEqual128);
        r.unary("negative", C8, C8, ComplexArithmetic::negative64);
        r.unary("negative", C16, C16, ComplexArithmetic::negative128);
        r.unary("positive", C8, C8, ComplexLoops::copy64);
        r.unary("positive", C16, C16, ComplexLoops::copy128);
        // conjugate is the identity on real numbers (NumPy has no bool loop)
        r.alias("conjugate", "copy", t -> !t.isBool() && !t.isComplex());
        r.unary("conjugate", C8, C8, ComplexArithmetic::conjugate64);
        r.unary("conjugate", C16, C16, ComplexArithmetic::conjugate128);
        r.unary("square", C8, C8, ComplexArithmetic::square64);
        r.unary("square", C16, C16, ComplexArithmetic::square128);
        r.unary("absolute", C8, DType.FLOAT32, ComplexArithmetic::absolute64);
        r.unary("absolute", C16, DType.FLOAT64, ComplexArithmetic::absolute128);
    }

    // ------------------------------------------------------------------ binary

    static void add64(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o, long po,
                      long so, long n) {
        if (sa == 0 && so == 0 && pa == po && a == o) {
            // reduction into one element: NumPy's pairwise summation of the run
            float re = a.get(F32, pa) + Pairwise.complex64Part(b, pb, sb, n);
            float im = a.get(F32, pa + 4) + Pairwise.complex64Part(b, pb + 4, sb, n);
            o.set(F32, po, re);
            o.set(F32, po + 4, im);
            return;
        }
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            float re = a.get(F32, pa) + b.get(F32, pb);
            float im = a.get(F32, pa + 4) + b.get(F32, pb + 4);
            o.set(F32, po, re);
            o.set(F32, po + 4, im);
        }
    }

    static void add128(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o, long po,
                       long so, long n) {
        if (sa == 0 && so == 0 && pa == po && a == o) {
            double re = a.get(F64, pa) + Pairwise.complex128Part(b, pb, sb, n);
            double im = a.get(F64, pa + 8) + Pairwise.complex128Part(b, pb + 8, sb, n);
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
            return;
        }
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            double re = a.get(F64, pa) + b.get(F64, pb);
            double im = a.get(F64, pa + 8) + b.get(F64, pb + 8);
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
        }
    }

    static void subtract64(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                           long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            float re = a.get(F32, pa) - b.get(F32, pb);
            float im = a.get(F32, pa + 4) - b.get(F32, pb + 4);
            o.set(F32, po, re);
            o.set(F32, po + 4, im);
        }
    }

    static void subtract128(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                            long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            double re = a.get(F64, pa) - b.get(F64, pb);
            double im = a.get(F64, pa + 8) - b.get(F64, pb + 8);
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
        }
    }

    static void multiply64(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                           long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            float ar = a.get(F32, pa);
            float ai = a.get(F32, pa + 4);
            float br = b.get(F32, pb);
            float bi = b.get(F32, pb + 4);
            o.set(F32, po, ar * br - ai * bi);
            o.set(F32, po + 4, ar * bi + ai * br);
        }
    }

    static void multiply128(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                            long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            double ar = a.get(F64, pa);
            double ai = a.get(F64, pa + 8);
            double br = b.get(F64, pb);
            double bi = b.get(F64, pb + 8);
            o.set(F64, po, ar * br - ai * bi);
            o.set(F64, po + 8, ar * bi + ai * br);
        }
    }

    static void divide64(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                         long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            float ar = a.get(F32, pa);
            float ai = a.get(F32, pa + 4);
            float br = b.get(F32, pb);
            float bi = b.get(F32, pb + 4);
            float brAbs = Math.abs(br);
            float biAbs = Math.abs(bi);
            float re;
            float im;
            if (brAbs >= biAbs) {
                if (brAbs == 0 && biAbs == 0) {
                    re = ar / brAbs;
                    im = ai / brAbs;
                } else {
                    float rat = bi / br;
                    float scl = 1.0f / (br + bi * rat);
                    re = (ar + ai * rat) * scl;
                    im = (ai - ar * rat) * scl;
                }
            } else {
                float rat = br / bi;
                float scl = 1.0f / (bi + br * rat);
                re = (ar * rat + ai) * scl;
                im = (ai * rat - ar) * scl;
            }
            o.set(F32, po, re);
            o.set(F32, po + 4, im);
        }
    }

    static void divide128(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                          long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            double ar = a.get(F64, pa);
            double ai = a.get(F64, pa + 8);
            double br = b.get(F64, pb);
            double bi = b.get(F64, pb + 8);
            double brAbs = Math.abs(br);
            double biAbs = Math.abs(bi);
            double re;
            double im;
            if (brAbs >= biAbs) {
                if (brAbs == 0 && biAbs == 0) {
                    re = ar / brAbs;
                    im = ai / brAbs;
                } else {
                    double rat = bi / br;
                    double scl = 1.0 / (br + bi * rat);
                    re = (ar + ai * rat) * scl;
                    im = (ai - ar * rat) * scl;
                }
            } else {
                double rat = br / bi;
                double scl = 1.0 / (bi + br * rat);
                re = (ar * rat + ai) * scl;
                im = (ai * rat - ar) * scl;
            }
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
        }
    }

    static void equal64(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                        long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            boolean eq = a.get(F32, pa) == b.get(F32, pb) && a.get(F32, pa + 4) == b.get(F32, pb + 4);
            o.set(I8, po, eq ? (byte) 1 : (byte) 0);
        }
    }

    static void equal128(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                         long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            boolean eq = a.get(F64, pa) == b.get(F64, pb) && a.get(F64, pa + 8) == b.get(F64, pb + 8);
            o.set(I8, po, eq ? (byte) 1 : (byte) 0);
        }
    }

    static void notEqual64(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                           long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            boolean ne = a.get(F32, pa) != b.get(F32, pb) || a.get(F32, pa + 4) != b.get(F32, pb + 4);
            o.set(I8, po, ne ? (byte) 1 : (byte) 0);
        }
    }

    static void notEqual128(MemorySegment a, long pa, long sa, MemorySegment b, long pb, long sb, MemorySegment o,
                            long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, pb += sb, po += so) {
            boolean ne = a.get(F64, pa) != b.get(F64, pb) || a.get(F64, pa + 8) != b.get(F64, pb + 8);
            o.set(I8, po, ne ? (byte) 1 : (byte) 0);
        }
    }

    // ------------------------------------------------------------------ unary

    static void negative64(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            float re = -a.get(F32, pa);
            float im = -a.get(F32, pa + 4);
            o.set(F32, po, re);
            o.set(F32, po + 4, im);
        }
    }

    static void negative128(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            double re = -a.get(F64, pa);
            double im = -a.get(F64, pa + 8);
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
        }
    }

    static void conjugate64(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            float re = a.get(F32, pa);
            float im = -a.get(F32, pa + 4);
            o.set(F32, po, re);
            o.set(F32, po + 4, im);
        }
    }

    static void conjugate128(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            double re = a.get(F64, pa);
            double im = -a.get(F64, pa + 8);
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
        }
    }

    static void square64(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            float re = a.get(F32, pa);
            float im = a.get(F32, pa + 4);
            o.set(F32, po, re * re - im * im);
            o.set(F32, po + 4, re * im + im * re);
        }
    }

    static void square128(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            double re = a.get(F64, pa);
            double im = a.get(F64, pa + 8);
            o.set(F64, po, re * re - im * im);
            o.set(F64, po + 8, re * im + im * re);
        }
    }

    static void absolute64(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            // Math.hypot of the widened parts, rounded once to float, as glibc's hypotf
            o.set(F32, po, (float) Math.hypot(a.get(F32, pa), a.get(F32, pa + 4)));
        }
    }

    static void absolute128(MemorySegment a, long pa, long sa, MemorySegment o, long po, long so, long n) {
        for (long i = 0; i < n; i++, pa += sa, po += so) {
            o.set(F64, po, Math.hypot(a.get(F64, pa), a.get(F64, pa + 8)));
        }
    }
}
