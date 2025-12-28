package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.DoubleRange;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;

/**
 * Complex arithmetic ufuncs. Expected values come from NumPy 2.5 unless stated otherwise.
 */
class ComplexUfuncTest {

    private static final double INF = Double.POSITIVE_INFINITY;
    private static final double NAN = Double.NaN;

    private static Complex c(double re, double im) {
        return Complex.of(re, im);
    }

    private static NDArray arr(Complex... values) {
        return NJ.array(values);
    }

    private static void assertValues(NDArray actual, Complex... expected) {
        assertEquals(expected.length, actual.size());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual.getComplex(i), "element " + i);
        }
    }

    @Test
    void addSubtractMultiply() {
        NDArray a = arr(c(1, 2), c(-3, 0.5), c(INF, 1));
        NDArray b = arr(c(0.5, -1), c(2, 2), c(INF, -INF));
        assertValues(NJ.add(a, b), c(1.5, 1), c(-1, 2.5), c(INF, -INF));
        assertValues(NJ.subtract(a, b), c(0.5, 3), c(-5, -1.5), c(NAN, INF));
        assertValues(NJ.multiply(a, b), c(2.5, 0), c(-7, -5), c(INF, NAN));
        // (inf + 0j) * 1j = nan + inf j with the textbook formula
        assertValues(NJ.multiply(arr(c(INF, 0)), arr(c(0, 1))), c(NAN, INF));
    }

    @Test
    void divideUsesSmithsAlgorithm() {
        NDArray a = arr(c(1, 2), c(1e300, 1e300), c(1, 1), c(-1, 0), c(0, 0), c(INF, 1), c(4, 6));
        NDArray b = arr(c(3, -4), c(1e300, 1e300), c(0, 0), c(0, 0), c(0, 0), c(1, 1), c(1, 2));
        // 1e300 / 1e300 does not overflow, unlike (a * conj(b)) / |b|^2
        assertValues(NJ.divide(a, b), c(-0.2, 0.4), c(1, 0), c(INF, INF), c(-INF, NAN), c(NAN, NAN),
                c(INF, -INF), c(3.2, -0.4));
        NDArray f = NJ.divide(NJ.array(DType.COMPLEX64, new Complex[]{c(1, 2)}),
                NJ.array(DType.COMPLEX64, new Complex[]{c(3, -4)}));
        assertEquals(DType.COMPLEX64, f.dtype());
        // NumPy: the float32 bit patterns 3192704204, 1053609164
        assertEquals(c(Float.intBitsToFloat((int) 3192704204L), Float.intBitsToFloat(1053609164)), f.getComplex(0));
    }

    @Test
    void unaryOperations() {
        NDArray a = arr(c(1, 2), c(-3, -0.5), c(0, -0.0));
        assertValues(NJ.negative(a), c(-1, -2), c(3, 0.5), c(-0.0, 0.0));
        assertValues(NJ.positive(a), c(1, 2), c(-3, -0.5), c(0, -0.0));
        assertValues(NJ.conjugate(a), c(1, -2), c(-3, 0.5), c(0, 0.0));
        assertValues(a.conj(), c(1, -2), c(-3, 0.5), c(0, 0.0));
        assertValues(NJ.square(a), c(-3, 4), c(8.75, 3), c(0, -0.0));
    }

    @Test
    void absoluteIsRealAndAvoidsOverflow() {
        double big = Math.scalb(1.0, 600);
        NDArray r = NJ.abs(arr(c(3 * big, 4 * big), c(INF, NAN), c(-0.0, -0.0), c(-5, 12)));
        assertEquals(DType.FLOAT64, r.dtype());
        assertEquals(5 * big, r.getDouble(0));
        assertEquals(INF, r.getDouble(1));
        assertEquals(0.0, r.getDouble(2));
        assertEquals(13.0, r.getDouble(3));
        NDArray f = NJ.abs(NJ.array(DType.COMPLEX64, new Complex[]{c(3, 4)}));
        assertEquals(DType.FLOAT32, f.dtype());
        assertEquals(5.0, f.getDouble(0));
    }

    @Test
    void conjugateOfRealArraysIsACopyInTheSameDtype() {
        NDArray i = NJ.array(DType.INT8, new double[]{1, -2});
        NDArray r = NJ.conjugate(i);
        assertEquals(DType.INT8, r.dtype());
        assertEquals(-2, r.getLong(1));
        assertEquals(DType.FLOAT32, NJ.conjugate(NJ.zeros(DType.FLOAT32, 2)).dtype());
        // bool has no loop of its own; as in NumPy, the first safe loop is int8
        assertEquals(DType.INT8, NJ.conjugate(NJ.zeros(DType.BOOL, 2)).dtype());
    }

    @Test
    void equalityComparesBothParts() {
        NDArray a = arr(c(1, 2), c(1, 2), c(NAN, 0), c(0.0, -0.0), c(3, 4));
        NDArray b = arr(c(1, 2), c(1, -2), c(NAN, 0), c(-0.0, 0.0), c(3, 5));
        NDArray eq = NJ.equal(a, b);
        NDArray ne = NJ.notEqual(a, b);
        boolean[] expected = {true, false, false, true, false};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], eq.getBoolean(i), "equal " + i);
            assertEquals(!expected[i], ne.getBoolean(i), "not_equal " + i);
        }
        // a real operand is promoted to complex
        assertTrue(NJ.equal(arr(c(2, 0)), NJ.array(new double[]{2})).getBoolean(0));
        assertFalse(NJ.equal(arr(c(2, 1e-300)), 2.0).getBoolean(0));
    }

    @Test
    void realOperandsArePromotedToComplex() {
        NDArray c64 = NJ.array(DType.COMPLEX64, new Complex[]{c(0, 1)});
        assertEquals(DType.COMPLEX64, NJ.add(NJ.array(new float[]{1.5f, 2}), c64).dtype());
        assertEquals(DType.COMPLEX128, NJ.add(NJ.array(new long[]{1, 2}), c64).dtype());
        assertEquals(DType.COMPLEX128, NJ.add(NJ.zeros(2), c64).dtype());
        assertValues(NJ.add(NJ.array(new long[]{1, 2}), c64), c(1, 1), c(2, 1));
        // weak Java scalars keep the complex dtype
        NDArray s = NJ.multiply(c64, 2.5);
        assertEquals(DType.COMPLEX64, s.dtype());
        assertEquals(c(0, 2.5), s.getComplex(0));
        assertEquals(c(3, 1), NJ.add(c64, 3L).getComplex(0));
        assertEquals(c(10, -10), Ufuncs.DIVIDE.apply(20.0, arr(c(1, 1))).getComplex(0));
    }

    @Test
    void reductionsAndOut() {
        NDArray a = arr(c(1, 2), c(3, -1), c(0.5, 0.25));
        assertEquals(c(4.5, 1.25), Ufuncs.ADD.reduce(a).getComplex());
        assertEquals(c(1.25, 3.75), Ufuncs.MULTIPLY.reduce(a).getComplex());
        assertEquals(Complex.ZERO, Ufuncs.ADD.reduce(NJ.zeros(DType.COMPLEX128, 0)).getComplex());
        assertEquals(Complex.ONE, Ufuncs.MULTIPLY.reduce(NJ.zeros(DType.COMPLEX64, 0)).getComplex());
        NDArray out = NJ.zeros(DType.COMPLEX128, 3);
        assertSame(out, Ufuncs.MULTIPLY.apply(a, a, out));
        assertEquals(c(-3, 4), out.getComplex(0));
        a.multiplyInPlace(2.0);
        assertEquals(c(6, -2), a.getComplex(1));
        // complex results do not fit a float out array (SAME_KIND)
        assertThrows(org.numjvm.exception.DTypeException.class, () -> Ufuncs.ADD.apply(a, a, NJ.zeros(3)));
    }

    @Test
    void stridedAndBroadcastOperands() {
        NDArray a = arr(c(1, 1), c(2, 2), c(3, 3), c(4, 4)).reshape(2, 2).transpose();
        NDArray row = arr(c(0, 1), c(1, 0));
        NDArray r = NJ.multiply(a, row);
        // a.T = [[1+1j, 3+3j], [2+2j, 4+4j]]
        assertEquals(c(-1, 1), r.getComplex(0, 0));
        assertEquals(c(3, 3), r.getComplex(0, 1));
        assertEquals(c(-2, 2), r.getComplex(1, 0));
        NDArray rev = arr(c(1, 0), c(2, 0), c(3, 0)).slice(Slice.step(-1));
        assertValues(NJ.subtract(rev, arr(c(0, 1))), c(3, -1), c(2, -1), c(1, -1));
    }

    @Test
    void realAndImagAreViews() {
        NDArray a = arr(c(1, 2), c(3, 4), c(5, 6));
        NDArray re = a.real();
        NDArray im = a.imag();
        assertEquals(DType.FLOAT64, re.dtype());
        assertEquals(3.0, re.getDouble(1));
        assertEquals(6.0, im.getDouble(2));
        assertEquals(16, re.strides().values()[0]);
        im.setDouble(0, -9);
        re.setDouble(2, 0.5);
        assertEquals(c(1, -9), a.getComplex(0));
        assertEquals(c(0.5, 6), a.getComplex(2));
        NDArray f = NJ.array(DType.COMPLEX64, new Complex[]{c(1, 2)}).imag();
        assertEquals(DType.FLOAT32, f.dtype());
        assertEquals(2.0, f.getDouble(0));
        // views of a view: the imaginary parts of a reversed, transposed array
        NDArray m = arr(c(1, 10), c(2, 20), c(3, 30), c(4, 40)).reshape(2, 2).transpose().slice(Slice.step(-1));
        // m = [[2+20j, 4+40j], [1+10j, 3+30j]]
        assertEquals(20.0, m.imag().getDouble(0, 0));
        assertEquals(30.0, m.imag().getDouble(1, 1));
        // read-only views stay read-only
        NDArray b = arr(c(1, 1)).broadcastTo(3);
        assertThrows(ReadOnlyArrayException.class, () -> b.real().setDouble(0, 1));
    }

    @Test
    void realAndImagOfRealArrays() {
        NDArray x = NJ.array(new double[]{1, 2});
        assertSame(x, x.real());
        NDArray im = x.imag();
        assertEquals(DType.FLOAT64, im.dtype());
        assertEquals(0.0, im.getDouble(1));
        assertFalse(im.isWriteable());
        assertThrows(ReadOnlyArrayException.class, () -> im.setDouble(0, 1));
        assertEquals(DType.INT8, NJ.zeros(DType.INT8, 2, 3).imag().dtype());
        assertEquals(2, NJ.zeros(DType.INT8, 2, 3).imag().ndim());
    }

    @Property(tries = 300)
    void divideInvertsMultiply(@ForAll @DoubleRange(min = -1e100, max = 1e100) double ar,
                               @ForAll @DoubleRange(min = -1e100, max = 1e100) double ai,
                               @ForAll @DoubleRange(min = 0.5, max = 1e3) double br,
                               @ForAll @DoubleRange(min = -1e3, max = 1e3) double bi) {
        NDArray a = arr(c(ar, ai));
        NDArray b = arr(c(br, bi));
        Complex q = NJ.divide(NJ.multiply(a, b), b).getComplex(0);
        double scale = Math.max(Math.abs(ar), Math.abs(ai)) + Double.MIN_NORMAL;
        assertTrue(Math.abs(q.re() - ar) <= 1e-12 * scale && Math.abs(q.im() - ai) <= 1e-12 * scale,
                () -> q + " vs " + c(ar, ai));
    }
}
