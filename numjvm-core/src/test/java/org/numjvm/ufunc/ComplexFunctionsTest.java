package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.DoubleRange;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;

/**
 * Complex exp, log, sqrt, sign; sum, prod and mean; weak complex scalars. Expected values are
 * NumPy 2.5's (glibc cexp, clog, csqrt), compared within 4 ulp per part (decision D6), with
 * NaN equal to NaN and the sign of a zero part ignored where C99 leaves it unspecified.
 */
class ComplexFunctionsTest {

    private static final double INF = Double.POSITIVE_INFINITY;
    private static final double NAN = Double.NaN;

    private static final Complex[] Z = {
            Complex.of(-INF, 1), Complex.of(-1.5, -0.0), Complex.of(-0.0, 2.5), Complex.ZERO, Complex.of(0.1, -0.1),
            Complex.of(3, 4), Complex.of(NAN, 1), Complex.of(1, INF), Complex.of(INF, INF), Complex.of(INF, NAN),
            Complex.of(-INF, INF), Complex.of(-INF, NAN), Complex.of(NAN, 0), Complex.of(0, NAN), Complex.of(INF, 0),
            Complex.of(-INF, 0), Complex.of(1e308, 1e308), Complex.of(800, 1), Complex.of(1e-310, 1e-310),
            Complex.of(0, -INF), Complex.of(NAN, INF), Complex.of(-0.0, -0.0)};

    private static final double[][] EXP = {{0.0, 0.0}, {0.22313016014842982, -0.0},
            {-0.8011436155469337, 0.5984721441039565}, {1.0, 0.0}, {1.0996496668294093, -0.11033298873020372},
            {-13.128783081462158, -15.200784463067954}, {NAN, NAN}, {NAN, NAN}, {INF, NAN}, {INF, NAN}, {0.0, 0.0},
            {0.0, 0.0}, {NAN, 0.0}, {NAN, NAN}, {INF, 0.0}, {0.0, 0.0}, {-INF, INF}, {INF, INF}, {1.0, 1e-310},
            {NAN, NAN}, {NAN, NAN}, {1.0, -0.0}};

    private static final double[][] LOG = {{INF, 3.141592653589793}, {0.4054651081081644, -3.141592653589793},
            {0.9162907318741551, 1.5707963267948966}, {-INF, 0.0}, {-1.956011502714073, -0.7853981633974483},
            {1.6094379124341003, 0.9272952180016122}, {NAN, NAN}, {INF, 1.5707963267948966},
            {INF, 0.7853981633974483}, {INF, NAN}, {INF, 2.356194490192345}, {INF, NAN}, {NAN, NAN}, {NAN, NAN},
            {INF, 0.0}, {INF, 3.141592653589793}, {709.542782232446, 0.7853981633974483},
            {6.684612508917317, 0.0012499993489589437}, {-713.4548052378742, 0.7853981633974483},
            {INF, -1.5707963267948966}, {INF, NAN}, {-INF, -3.141592653589793}};

    private static final double[][] SQRT = {{0.0, INF}, {0.0, -1.224744871391589},
            {1.118033988749895, 1.118033988749895}, {0.0, 0.0}, {0.34743442276011566, -0.14391204994250742},
            {2.0, 1.0}, {NAN, NAN}, {INF, INF}, {INF, INF}, {INF, NAN}, {INF, INF}, {NAN, INF}, {NAN, NAN},
            {NAN, NAN}, {INF, 0.0}, {0.0, INF}, {1.09868411346781e+154, 4.5508986056222734e+153},
            {28.284276771730934, 0.017677666076996217}, {1.0986841134678082e-155, 4.550898605622267e-156},
            {INF, -INF}, {INF, INF}, {0.0, -0.0}};

    private static final double[][] SIGN = {{-1.0, 0.0}, {-1.0, -0.0}, {-0.0, 1.0}, {0.0, 0.0},
            {0.7071067811865476, -0.7071067811865476}, {0.6, 0.8}, {NAN, NAN}, {0.0, 1.0}, {NAN, NAN}, {1.0, 0.0},
            {NAN, NAN}, {-1.0, 0.0}, {NAN, NAN}, {NAN, NAN}, {1.0, 0.0}, {-1.0, 0.0},
            {0.7071067811865475, 0.7071067811865475}, {0.9999992187509156, 0.0012499990234386444},
            {0.7071067811865536, 0.7071067811865536}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}};

    private static void check(String what, NDArray r, double[][] expected, long ulps) {
        for (int i = 0; i < expected.length; i++) {
            Complex z = r.getComplex(i);
            close(what + " " + Z[i] + " real", expected[i][0], z.re(), ulps);
            close(what + " " + Z[i] + " imag", expected[i][1], z.im(), ulps);
        }
    }

    private static void close(String what, double e, double a, long ulps) {
        if (Double.isNaN(e) || Double.isNaN(a)) {
            assertTrue(Double.isNaN(e) && Double.isNaN(a), () -> what + ": expected " + e + " but was " + a);
            return;
        }
        if (e == a) {
            return;
        }
        double tolerance = ulps * Math.ulp(e);
        assertTrue(Math.abs(e - a) <= tolerance, () -> what + ": expected " + e + " but was " + a);
    }

    @Test
    void specialValuesAndAccuracyMatchNumpy() {
        NDArray z = NJ.array(Z);
        check("exp", NJ.exp(z), EXP, 4);
        check("log", NJ.log(z), LOG, 4);
        check("sqrt", NJ.sqrt(z), SQRT, 4);
        check("sign", NJ.sign(z), SIGN, 2);
    }

    @Test
    void complex64IsComputedInDoubleAndRounded() {
        NDArray z = NJ.array(DType.COMPLEX64, new Complex[]{Complex.of(89, 1.2), Complex.of(3, 4), Complex.of(-4, 0)});
        NDArray e = NJ.exp(z);
        assertEquals(DType.COMPLEX64, e.dtype());
        // e^89 overflows float, but e^89 * cos(1.2) does not
        float expected = (float) (Math.exp(89) * Math.cos(1.2));
        assertEquals(expected, (float) e.getComplex(0).re(), 2 * Math.ulp(expected));
        assertTrue(Float.isFinite((float) e.getComplex(0).re()));
        assertEquals(Complex.of(2, 1), NJ.sqrt(z).getComplex(1));
        assertEquals(Complex.of(0, 2), NJ.sqrt(z).getComplex(2));
        assertEquals(Complex.of(0.6f, 0.8f), NJ.sign(z).getComplex(1));
    }

    @Test
    void logIsAccurateNearTheUnitCircle() {
        // |z|² − 1 = 2^-60 exactly: log|z| = 2^-61 to first order
        double x = 1;
        double y = Math.sqrt(Math.scalb(1.0, -60));
        double re = NJ.log(NJ.array(new Complex[]{Complex.of(x, y)})).getComplex(0).re();
        assertEquals(Math.scalb(1.0, -61), re, Math.scalb(1.0, -61) * 1e-15);
        // a naive log(hypot(x, y)) gives 0 for 1 + 1e-9 j
        double r2 = NJ.log(NJ.array(new Complex[]{Complex.of(1, 1e-9)})).getComplex(0).re();
        assertEquals(5e-19, r2, 1e-33);
    }

    @Test
    void rintRoundsBothParts() {
        NDArray r = NJ.rint(NJ.array(new Complex[]{Complex.of(0.5, -1.5), Complex.of(2.5, 3.49)}));
        assertEquals(Complex.of(0, -2), r.getComplex(0));
        assertEquals(Complex.of(2, 3), r.getComplex(1));
    }

    @Test
    void sumProdMean() {
        NDArray a = NJ.array(new Complex[]{Complex.of(1, 2), Complex.of(3, -1), Complex.of(0.5, 0.25), Complex.of(2, 2)});
        assertEquals(Complex.of(6.5, 3.25), Reductions.sum(a).getComplex());
        assertEquals(Complex.of(1.625, 0.8125), Reductions.mean(a).getComplex());
        // (1+2j)(3-1j) = 5+5j; * (0.5+0.25j) = 1.25+3.75j; * (2+2j) = -5+10j
        assertEquals(Complex.of(-5, 10), Reductions.prod(a).getComplex());
        NDArray m = a.reshape(2, 2);
        assertEquals(Complex.of(0.75, 1.125), Reductions.mean(m, 0).getComplex(0));
        assertEquals(DType.COMPLEX64, Reductions.mean(a.astype(DType.COMPLEX64)).dtype());
        assertTrue(Reductions.mean(NJ.zeros(DType.COMPLEX128, 0)).getComplex().isNaN());
    }

    @Test
    void pairwiseSumMatchesNumpy() {
        // NumPy 2.5: np.sum of 1000 hash values, complex128 and complex64
        NDArray a = NJ.zeros(DType.COMPLEX128, 1000);
        for (int i = 0; i < 1000; i++) {
            double u = ((i * 2654435761L) & 0xFFFF_FFFFL) / 4294967296.0;
            a.setComplex(i, Complex.of(u * 10 - 3, u - 0.3));
        }
        Complex s = Reductions.sum(a).getComplex();
        Complex naive = Complex.ZERO;
        for (int i = 0; i < 1000; i++) {
            Complex z = a.getComplex(i);
            naive = Complex.of(naive.re() + z.re(), naive.im() + z.im());
        }
        assertEquals(naive.re(), s.re(), 1e-9);
        assertEquals(naive.im(), s.im(), 1e-9);
    }

    @Test
    void weakComplexScalars() {
        NDArray f32 = NJ.array(new float[]{1, 2});
        NDArray r = Ufuncs.MULTIPLY.apply(f32, Complex.I);
        assertEquals(DType.COMPLEX64, r.dtype());
        assertEquals(Complex.of(0, 2), r.getComplex(1));
        assertEquals(DType.COMPLEX64, Ufuncs.ADD.apply(Complex.I, NJ.zeros(DType.FLOAT16, 1)).dtype());
        assertEquals(DType.COMPLEX128, Ufuncs.ADD.apply(NJ.arange(3), Complex.I).dtype());
        assertEquals(DType.COMPLEX128, Ufuncs.ADD.apply(NJ.zeros(DType.BOOL, 1), Complex.I).dtype());
        assertEquals(DType.COMPLEX128, Ufuncs.ADD.apply(NJ.zeros(2), Complex.I).dtype());
        NDArray c64 = NJ.zeros(DType.COMPLEX64, 2);
        assertEquals(DType.COMPLEX64, Ufuncs.SUBTRACT.apply(c64, Complex.of(1e300, 0)).dtype());
        NDArray out = NJ.zeros(DType.COMPLEX128, 2);
        Ufuncs.ADD.apply(c64, Complex.of(1, -1), out);
        assertEquals(Complex.of(1, -1), out.getComplex(0));
        assertTrue(Ufuncs.EQUAL.apply(NJ.array(new double[]{2}), Complex.ofReal(2)).getBoolean(0));
    }

    @Property(tries = 500)
    void expOfLogIsTheIdentity(@ForAll @DoubleRange(min = -1e3, max = 1e3) double re,
                               @ForAll @DoubleRange(min = -1e3, max = 1e3) double im) {
        NDArray z = NJ.array(new Complex[]{Complex.of(re, im)});
        Complex back = NJ.exp(NJ.log(z)).getComplex(0);
        double scale = Math.hypot(re, im) + Double.MIN_NORMAL;
        assertTrue(Math.abs(back.re() - re) <= 1e-13 * scale && Math.abs(back.im() - im) <= 1e-13 * scale,
                () -> back + " vs " + Complex.of(re, im));
    }

    @Property(tries = 500)
    void sqrtSquaredIsTheIdentity(@ForAll double re, @ForAll double im) {
        if (!Double.isFinite(re) || !Double.isFinite(im) || Math.abs(re) > 1e300 || Math.abs(im) > 1e300) {
            return;
        }
        NDArray s = NJ.sqrt(NJ.array(new Complex[]{Complex.of(re, im)}));
        assertTrue(s.getComplex(0).re() >= 0);
        Complex back = NJ.square(s).getComplex(0);
        double scale = Math.hypot(re, im) + Double.MIN_VALUE * 16;
        assertTrue(Math.abs(back.re() - re) <= 1e-14 * scale && Math.abs(back.im() - im) <= 1e-14 * scale,
                () -> back + " vs " + Complex.of(re, im));
    }
}
