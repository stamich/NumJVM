package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.memory.NJScope;

/**
 * Expected strings are NumPy 2 {@code repr} output for the same arrays
 * ({@code NUMPY_COMPATIBILITY}).
 */
class FormatTest {

    private static NDArray reshape(NDArray a, long... shape) {
        return SharedSecrets.arrayAccess().create(SharedSecrets.arrayAccess().buffer(a), a.dtype(), 0,
                Shape.of(shape), Strides.cContiguous(Shape.of(shape), a.itemSize()), true);
    }

    @Test
    void floats() {
        assertEquals("array([1., 2., 3.])", NJ.array(new double[]{1, 2, 3}).toString());
        assertEquals("array([0.1 , 0.25])", NJ.array(new double[]{0.1, 0.25}).toString());
        assertEquals("array([ 1.5 , 10.25])", NJ.array(new double[]{1.5, 10.25}).toString());
        assertEquals("array([-0.5,  2. ])", NJ.array(new double[]{-0.5, 2.0}).toString());
        assertEquals("array([0.33333333, 0.66666667])", NJ.array(new double[]{1.0 / 3, 2.0 / 3}).toString());
        assertEquals("array([1000000., 1000001., 1000002., 1000003., 1000004.])",
                NJ.arange(1e6, 1e6 + 5, 1.0).toString());
        assertEquals("array([-0.,  1.])", NJ.array(new double[]{-0.0, 1.0}).toString());
    }

    @Test
    void scientificNotation() {
        assertEquals("array([1.e-05, 1.e+10])", NJ.array(new double[]{1e-5, 1e10}).toString());
        assertEquals("array([1.e-09, 2.e-09])", NJ.array(new double[]{1e-9, 2e-9}).toString());
    }

    @Test
    void specialValues() {
        assertEquals("array([ nan,  inf, -1.5])",
                NJ.array(new double[]{Double.NaN, Double.POSITIVE_INFINITY, -1.5}).toString());
        assertEquals("array([  3., -inf])", NJ.array(new double[]{3.0, Double.NEGATIVE_INFINITY}).toString());
        assertEquals("array([[nan]])", NJ.array(new double[]{Double.NaN}, 1, 1).toString());
    }

    @Test
    void otherDTypes() {
        assertEquals("array([  1,   2, 300])", NJ.array(new long[]{1, 2, 300}).toString());
        assertEquals("array([  -1,   20, -300])", NJ.array(new long[]{-1, 20, -300}).toString());
        assertEquals("array([1, 2], dtype=int32)", NJ.array(new int[]{1, 2}).toString());
        assertEquals("array([ True, False])", NJ.array(new boolean[]{true, false}).toString());
        assertEquals("array([0.1], dtype=float32)", NJ.array(new float[]{0.1f}).toString());
        assertEquals("array([ 1.25, -2.5 ], dtype=float32)", NJ.array(new float[]{1.25f, -2.5f}).toString());
        assertEquals("array([0.1], dtype=float16)", NJ.array(DType.FLOAT16, new double[]{0.1}).toString());
        assertEquals("array([18446744073709551615], dtype=uint64)",
                NJ.array(new long[]{-1}).astype(DType.UINT64).toString());
        assertEquals("array([1.+0.j, 2.+0.j])", NJ.array(new double[]{1, 2}).astype(DType.COMPLEX128).toString());
    }

    /**
     * Expected strings from NumPy 2.5 {@code repr}: real and imaginary parts are aligned as
     * separate columns, the imaginary part always signed.
     */
    @Test
    void complexArrays() {
        assertEquals("array([1. +2.j, 3.5-1.j])",
                NJ.array(new Complex[]{c(1, 2), c(3.5, -1)}).toString());
        assertEquals("array([1. +2.25j, 3.5-1.j  ])",
                NJ.array(new Complex[]{c(1, 2.25), c(3.5, -1)}).toString());
        assertEquals("array([1.e+09+1.e+00j, 1.e+00-1.e-06j])",
                NJ.array(new Complex[]{c(1e9, 1), c(1, -1e-6)}).toString());
        assertEquals("array([ nan +1.j,   1.+infj, -inf -2.j])",
                NJ.array(new Complex[]{c(Double.NaN, 1), c(1, Double.POSITIVE_INFINITY), c(Double.NEGATIVE_INFINITY, -2)}).toString());
        assertEquals("array([-1.  -1.j, 10.+100.j])",
                NJ.array(new Complex[]{c(-1, -1), c(10, 100)}).toString());
        assertEquals("array([1.+1.e+10j])",
                NJ.array(new Complex[]{c(1, 1e10)}).toString());
        assertEquals("array([ 0.+nanj, -0. -0.j])",
                NJ.array(new Complex[]{c(0, Double.NaN), c(-0.0, -0.0)}).toString());
        assertEquals("array([[1.+1.j, 2.+0.j],\n       [3.+0.j, 0.+4.j]])",
                NJ.array(new Complex[]{c(1, 1), c(2, 0), c(3, 0), c(0, 4)}, 2, 2).toString());
        assertEquals("array([1. +2.j, 3.5-1.j], dtype=complex64)",
                NJ.array(DType.COMPLEX64, new Complex[]{c(1, 2), c(3.5, -1)}).toString());
        assertEquals("array([0.1+0.2j], dtype=complex64)",
                NJ.array(DType.COMPLEX64, new Complex[]{c(0.1, 0.2)}).toString());
        assertEquals("array(1.5-2.j)",
                NJ.scalar(c(1.5, -2)).toString());
        assertEquals("array([0.000e+00+0.j, 1.000e+00+0.j, 2.000e+00+0.j, ..., 1.997e+03+0.j,\n       1.998e+03+0.j, 1.999e+03+0.j], shape=(2000,))",
                NJ.arange(2000).astype(DType.COMPLEX128).toString());
    }

    private static Complex c(double re, double im) {
        return Complex.of(re, im);
    }

    @Test
    void scalarsAndEmptyArrays() {
        assertEquals("array(2.5)", NJ.scalar(2.5).toString());
        assertEquals("array(7)", NJ.scalar(7L).toString());
        assertEquals("array([], dtype=float64)", NJ.zeros(0).toString());
        assertEquals("array([], shape=(0, 3), dtype=float64)", NJ.zeros(0, 3).toString());
        assertEquals("array([], shape=(2, 0, 3), dtype=int32)", NJ.zeros(DType.INT32, 2, 0, 3).toString());
    }

    @Test
    void multiDimensional() {
        assertEquals("""
                array([[0., 1., 2.],
                       [3., 4., 5.]])""", reshape(NJ.arange(0.0, 6.0, 1.0), 2, 3).toString());
        assertEquals("""
                array([[1.5 , 2.25],
                       [3.  , 4.  ]])""", NJ.array(new double[]{1.5, 2.25, 3, 4}, 2, 2).toString());
        assertEquals("""
                array([[[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]],
                
                       [[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]]])""", reshape(NJ.arange(24), 2, 3, 4).toString());
    }

    @Test
    void longRowsWrapAt75Characters() {
        assertEquals("""
                array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
                       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
                       26., 27., 28., 29.])""", NJ.arange(0.0, 30.0, 1.0).toString());
    }

    @Test
    void largeArraysAreSummarised() {
        assertEquals("array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))", NJ.arange(2000).toString());
        assertEquals("""
                        array([[0.000e+00, 1.000e+00, 2.000e+00, ..., 4.700e+01, 4.800e+01,
                                4.900e+01],
                               [5.000e+01, 5.100e+01, 5.200e+01, ..., 9.700e+01, 9.800e+01,
                                9.900e+01],
                               [1.000e+02, 1.010e+02, 1.020e+02, ..., 1.470e+02, 1.480e+02,
                                1.490e+02],
                               ...,
                               [1.850e+03, 1.851e+03, 1.852e+03, ..., 1.897e+03, 1.898e+03,
                                1.899e+03],
                               [1.900e+03, 1.901e+03, 1.902e+03, ..., 1.947e+03, 1.948e+03,
                                1.949e+03],
                               [1.950e+03, 1.951e+03, 1.952e+03, ..., 1.997e+03, 1.998e+03,
                                1.999e+03]], shape=(40, 50))""",
                reshape(NJ.arange(0.0, 2000.0, 1.0), 40, 50).toString());
        String big3d = reshape(NJ.arange(3000), 10, 15, 20).toString();
        assertTrue(big3d.startsWith("array([[[   0,    1,    2, ...,   17,   18,   19],"), big3d);
        assertTrue(big3d.contains("\n\n       ...,\n\n       [[2100,"), big3d);
        assertTrue(big3d.endsWith("[2980, 2981, 2982, ..., 2997, 2998, 2999]]], shape=(10, 15, 20))"), big3d);
    }

    @Test
    void inaccessibleArraysStillPrint() {
        NDArray a;
        try (NJScope scope = NJ.scope()) {
            a = scope.zeros(DType.FLOAT32, 2, 3);
        }
        String s = a.toString();
        assertTrue(s.startsWith("array(<inaccessible: confined NJScope#"), s);
        assertTrue(s.endsWith("is closed>, shape=[2, 3], dtype=float32)"), s);
    }
}
