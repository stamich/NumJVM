package org.numjvm.internal.loops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.numjvm.dtype.DType;

/**
 * Complex exp, log, sqrt, sign and rint loops for complex64 and complex128.
 *
 * <p>Special values follow C99 Annex G, as the C library functions NumPy calls
 * ({@code cexp}, {@code clog}, {@code csqrt}); sign is NumPy 2's {@code z / |z|}. Finite
 * results are accurate to a few units in the last place of each part (decision D6): exp is
 * {@code e^a (cos b + i sin b)}, scaled so that it overflows only when the result does; the
 * real part of log is {@code log1p} of {@code |z|² − 1}, evaluated in double-double arithmetic,
 * near the unit circle and {@code log |z|} elsewhere; sqrt is algorithm 312 (Friedland), with
 * scaling of very large and very small arguments. complex64 values are computed in
 * {@code double} and each part rounded once to {@code float}.
 */
final class ComplexMath {

    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED;
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED;

    private static final double LN2 = 0.6931471805599453;
    /** {@code 1000 · ln 2}, correctly rounded. */
    private static final double LN2_1000 = 693.1471805599453;
    private static final double TWO_P1000 = 0x1p1000;
    /** Above this, {@code a + hypot(a, b)} in sqrt could overflow (FreeBSD's {@code THRESH}). */
    private static final double SQRT_THRESH = 0x1.a827999fcef32p+1022;
    /** Largest {@code a} for which {@code exp(a)} is finite. */
    private static final double EXP_MAX = 709.782712893384;

    private ComplexMath() {
    }

    /**
     * A function of one complex argument that writes its result.
     */
    @FunctionalInterface
    interface Op {
        void apply(double a, double b, MemorySegment o, long po, boolean single);
    }

    static void register(KernelRegistry.Registrar r) {
        for (Object[] f : new Object[][]{
                {"exp", (Op) ComplexMath::exp}, {"log", (Op) ComplexMath::log}, {"sqrt", (Op) ComplexMath::sqrt},
                {"sign", (Op) ComplexMath::sign}, {"rint", (Op) ComplexMath::rint}}) {
            String name = (String) f[0];
            Op op = (Op) f[1];
            r.unary(name, DType.COMPLEX64, DType.COMPLEX64, loop64(op));
            r.unary(name, DType.COMPLEX128, DType.COMPLEX128, loop128(op));
        }
    }

    private static UnaryLoop loop64(Op op) {
        return (in, pi, si, out, po, so, n) -> {
            for (long i = 0; i < n; i++, pi += si, po += so) {
                op.apply(in.get(F32, pi), in.get(F32, pi + 4), out, po, true);
            }
        };
    }

    private static UnaryLoop loop128(Op op) {
        return (in, pi, si, out, po, so, n) -> {
            for (long i = 0; i < n; i++, pi += si, po += so) {
                op.apply(in.get(F64, pi), in.get(F64, pi + 8), out, po, false);
            }
        };
    }

    private static void write(MemorySegment o, long po, boolean single, double re, double im) {
        if (single) {
            o.set(F32, po, (float) re);
            o.set(F32, po + 4, (float) im);
        } else {
            o.set(F64, po, re);
            o.set(F64, po + 8, im);
        }
    }

    // ------------------------------------------------------------------ exp

    static void exp(double a, double b, MemorySegment o, long po, boolean single) {
        if (Double.isNaN(a)) {
            write(o, po, single, Double.NaN, b == 0 ? b : Double.NaN);
        } else if (Double.isInfinite(a)) {
            if (a > 0) {
                if (b == 0) {
                    write(o, po, single, a, b);
                } else if (Double.isFinite(b)) {
                    write(o, po, single, Math.copySign(a, Math.cos(b)), Math.copySign(a, Math.sin(b)));
                } else {
                    write(o, po, single, a, Double.NaN);
                }
            } else if (Double.isFinite(b)) {
                write(o, po, single, Math.copySign(0.0, Math.cos(b)), Math.copySign(0.0, Math.sin(b)));
            } else {
                write(o, po, single, 0.0, 0.0);
            }
        } else if (!Double.isFinite(b)) {
            write(o, po, single, Double.NaN, Double.NaN);
        } else if (b == 0) {
            write(o, po, single, Math.exp(a), b);
        } else if (a > EXP_MAX) {
            // e^a overflows on its own; e^(a/2) twice keeps cos and sin in range
            double h = Math.exp(a * 0.5);
            write(o, po, single, h * Math.cos(b) * h, h * Math.sin(b) * h);
        } else {
            double e = Math.exp(a);
            write(o, po, single, e * Math.cos(b), e * Math.sin(b));
        }
    }

    // ------------------------------------------------------------------ log

    static void log(double a, double b, MemorySegment o, long po, boolean single) {
        if (Double.isNaN(a) || Double.isNaN(b)) {
            boolean inf = Double.isInfinite(a) || Double.isInfinite(b);
            write(o, po, single, inf ? Double.POSITIVE_INFINITY : Double.NaN, Double.NaN);
            return;
        }
        write(o, po, single, logAbs(a, b), Math.atan2(b, a));
    }

    /**
     * {@code log |a + bi|} without overflow, underflow or cancellation near {@code |z| = 1}.
     */
    static double logAbs(double a, double b) {
        double x = Math.abs(a);
        double y = Math.abs(b);
        if (x < y) {
            double t = x;
            x = y;
            y = t;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return x;
        }
        if (x == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        if (x > 0x1p1000) {
            return Math.log(Math.hypot(x * 0.5, y * 0.5)) + LN2;
        }
        if (x < 0x1p-1000) {
            return Math.log(Math.hypot(x * TWO_P1000, y * TWO_P1000)) - LN2_1000;
        }
        double h = Math.hypot(x, y);
        if (h < 0.5 || h > 2) {
            return Math.log(h);
        }
        // x² + y² − 1 in double-double: exact products, then error-free sums
        double x2 = x * x;
        double x2e = Math.fma(x, x, -x2);
        double y2 = y * y;
        double y2e = Math.fma(y, y, -y2);
        double s1 = x2 - 1.0;
        double e1 = (x2 - (s1 + 1.0)) + (-1.0 - (s1 - x2));
        double s2 = s1 + y2;
        double bb = s2 - s1;
        double e2 = (s1 - (s2 - bb)) + (y2 - bb);
        double t = s2 + (e1 + e2 + x2e + y2e);
        return 0.5 * Math.log1p(t);
    }

    // ------------------------------------------------------------------ sqrt

    static void sqrt(double a, double b, MemorySegment o, long po, boolean single) {
        if (a == 0 && b == 0) {
            write(o, po, single, 0.0, b);
            return;
        }
        if (Double.isInfinite(b)) {
            write(o, po, single, Double.POSITIVE_INFINITY, b);
            return;
        }
        if (Double.isNaN(a)) {
            write(o, po, single, a, Double.NaN);
            return;
        }
        if (Double.isInfinite(a)) {
            // C99: csqrt(-inf + bi) = 0 + inf·i and csqrt(+inf + bi) = inf + 0i for finite b,
            // with NaN in place of the 0 when b is NaN
            double zero = Double.isNaN(b) ? Double.NaN : 0.0;
            if (a < 0) {
                write(o, po, single, zero, Math.copySign(a, b));
            } else {
                write(o, po, single, a, Math.copySign(zero, b));
            }
            return;
        }
        double scale = 1;
        if (Math.abs(a) >= SQRT_THRESH || Math.abs(b) >= SQRT_THRESH) {
            a *= 0.25;
            b *= 0.25;
            scale = 2;
        } else if (Math.abs(a) < 0x1p-1000 && Math.abs(b) < 0x1p-1000) {
            a *= TWO_P1000;
            b *= TWO_P1000;
            scale = 0x1p-500;
        }
        double re;
        double im;
        if (a >= 0) {
            double t = Math.sqrt((a + Math.hypot(a, b)) * 0.5);
            re = t;
            im = b / (2 * t);
        } else {
            double t = Math.sqrt((-a + Math.hypot(a, b)) * 0.5);
            re = Math.abs(b) / (2 * t);
            im = Math.copySign(t, b);
        }
        write(o, po, single, re * scale, im * scale);
    }

    // ------------------------------------------------------------------ sign and rint

    /**
     * NumPy 2's complex sign: {@code z / |z|}, with ±1 or ±i for infinite arguments and 0 for 0.
     */
    static void sign(double a, double b, MemorySegment o, long po, boolean single) {
        double abs = single ? (float) Math.hypot(a, b) : Math.hypot(a, b);
        if (Double.isNaN(abs)) {
            write(o, po, single, Double.NaN, Double.NaN);
        } else if (Double.isInfinite(abs)) {
            if (Double.isInfinite(a)) {
                if (Double.isInfinite(b)) {
                    write(o, po, single, Double.NaN, Double.NaN);
                } else {
                    write(o, po, single, Math.copySign(1.0, a), 0.0);
                }
            } else {
                write(o, po, single, 0.0, Math.copySign(1.0, b));
            }
        } else if (abs == 0) {
            write(o, po, single, 0.0, 0.0);
        } else if (single) {
            write(o, po, true, (float) a / (float) abs, (float) b / (float) abs);
        } else {
            write(o, po, false, a / abs, b / abs);
        }
    }

    /**
     * Rounds each part to the nearest integer, halves to even.
     */
    static void rint(double a, double b, MemorySegment o, long po, boolean single) {
        write(o, po, single, Math.rint(a), Math.rint(b));
    }
}
