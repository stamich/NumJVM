package org.numjvm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ShapeException;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;

class FactoriesTest {

    @Test
    void zerosForScalarEmptyAndNdShapes() {
        NDArray s = NJ.zeros(DType.FLOAT32);
        assertEquals(Shape.scalar(), s.shape());
        assertEquals(1, s.size());
        assertEquals(0.0, s.getDouble());

        NDArray e = NJ.zeros(DType.INT16, 0, 4);
        assertEquals(0, e.size());
        assertEquals(Strides.of(8, 2), e.strides());

        NDArray m = NJ.zeros(2, 3);
        assertEquals(DType.FLOAT64, m.dtype());
        assertEquals(Strides.of(24, 8), m.strides());
        assertTrue(m.isCContiguous());
        assertTrue(m.isWriteable());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"COMPLEX64", "COMPLEX128"}, mode = EnumSource.Mode.EXCLUDE)
    void onesAndFullForEveryRealDType(DType dtype) {
        NDArray o = NJ.ones(dtype, 3, 5);
        NDArray f = NJ.full(dtype, 2.0, 7);
        for (long i = 0; i < 3; i++) {
            for (long j = 0; j < 5; j++) {
                assertEquals(1.0, o.getDouble(i, j));
            }
        }
        for (long i = 0; i < 7; i++) {
            assertEquals(dtype == DType.BOOL ? 1.0 : 2.0, f.getDouble(i));
        }
    }

    @Test
    void fullReplicatesNegativeZeroAndLargeArrays() {
        NDArray z = NJ.full(DType.FLOAT64, -0.0, 3);
        assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(z.getDouble(2)));
        NDArray big = NJ.full(DType.FLOAT32, 0.5, 1000, 1001);
        assertEquals(0.5, big.getDouble(999, 1000));
        assertEquals(0.5, big.getDouble(0, 0));
        assertEquals(0, NJ.full(DType.INT8, 9, 0).size());
    }

    @Test
    void complexFactories() {
        assertEquals(Complex.ONE, NJ.ones(DType.COMPLEX128, 2).getComplex(1));
        assertEquals(Complex.of(2.5, 0), NJ.full(DType.COMPLEX64, 2.5, 3).getComplex(2));
        NDArray a = NJ.array(new Complex[]{Complex.of(1, 2), Complex.of(3, -4), Complex.I, Complex.ZERO}, 2, 2);
        assertEquals(DType.COMPLEX128, a.dtype());
        assertEquals(Complex.of(3, -4), a.getComplex(0, 1));
        NDArray b = NJ.array(DType.COMPLEX64, new Complex[]{Complex.of(0.1, 1e40)});
        assertEquals(Complex.of((float) 0.1, Double.POSITIVE_INFINITY), b.getComplex(0));
        assertThrows(DTypeException.class, () -> NJ.array(DType.FLOAT64, new Complex[]{Complex.ONE}));
        assertEquals(Complex.of(-1, 1), NJ.scalar(Complex.of(-1, 1)).getComplex());
        try (var scope = NJ.scope()) {
            assertEquals(Complex.I, scope.array(new Complex[]{Complex.I}).getComplex(0));
        }
    }

    @Test
    void emptyIsZeroFilled() {
        assertEquals(0L, NJ.empty(DType.INT64, 10).getLong(9));
    }

    @Test
    void arrayFromJavaArrays() {
        NDArray d = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(Shape.of(2, 3), d.shape());
        assertEquals(6.0, d.getDouble(1, 2));

        NDArray f = NJ.array(new float[]{1.5f, 2.5f});
        assertEquals(DType.FLOAT32, f.dtype());
        assertEquals(Shape.of(2), f.shape());

        NDArray l = NJ.array(new long[]{Long.MAX_VALUE, -1});
        assertEquals(DType.INT64, l.dtype());
        assertEquals(Long.MAX_VALUE, l.getLong(0));

        NDArray i = NJ.array(new int[]{7, 8, 9}, 3, 1);
        assertEquals(DType.INT32, i.dtype());
        assertEquals(9, i.getLong(2, 0));

        NDArray b = NJ.array(new boolean[]{true, false});
        assertEquals(DType.BOOL, b.dtype());
        assertTrue(b.getBoolean(0));
        assertFalse(b.getBoolean(1));

        NDArray u = NJ.array(DType.UINT8, new double[]{0, 255, 256}, 3);
        assertEquals(255, u.getLong(1));
        assertEquals(0, u.getLong(2));
    }

    @Test
    void arrayValuesAreCopied() {
        double[] values = {1, 2};
        NDArray a = NJ.array(values);
        values[0] = 99;
        assertEquals(1.0, a.getDouble(0));
    }

    @Test
    void shapeMustMatchTheNumberOfValues() {
        ShapeException e = assertThrows(ShapeException.class, () -> NJ.array(new double[]{1, 2, 3}, 2, 2));
        assertTrue(e.getMessage().contains("3 values cannot fill shape [2, 2]"), e.getMessage());
        assertEquals(Shape.of(1), NJ.array(new double[]{5}).shape(), "no shape means 1-D");
    }

    @Test
    void scalars() {
        NDArray d = NJ.scalar(2.5);
        assertEquals(0, d.ndim());
        assertEquals(2.5, d.getDouble());
        NDArray l = NJ.scalar(7L);
        assertEquals(DType.INT64, l.dtype());
        assertEquals(7L, l.getLong());
    }

    @Test
    void integerArange() {
        NDArray a = NJ.arange(0, 12, 1);
        assertEquals(DType.INT64, a.dtype());
        assertEquals(12, a.size());
        assertEquals(11, a.getLong(11));
        assertEquals(5, NJ.arange(5).size());
        NDArray down = NJ.arange(10, 0, -3);
        assertEquals(4, down.size());
        assertEquals(1, down.getLong(3));
        assertEquals(0, NJ.arange(5, 5, 1).size());
        assertEquals(0, NJ.arange(5, 0, 1).size());
        assertThrows(ShapeException.class, () -> NJ.arange(0, 10, 0));
    }

    @Test
    void floatArangeFollowsNumPyLength() {
        NDArray a = NJ.arange(0.0, 1.0, 0.1);
        assertEquals(10, a.size());               // np.arange(0, 1, 0.1).size
        assertEquals(0.30000000000000004, a.getDouble(3));
        assertEquals(4, NJ.arange(1.0, 2.0, 0.3).size());
        assertEquals(0, NJ.arange(1.0, 0.0, 0.5).size());
        assertThrows(ShapeException.class, () -> NJ.arange(0.0, 1.0, 0.0));
        assertThrows(ShapeException.class, () -> NJ.arange(0.0, Double.NaN, 1.0));
    }

    @Test
    void largeArraysGoOffHeapSmallOnesStayOnTheHeap() {
        assertFalse(NJ.zeros(DType.FLOAT64, 1000).isNative());
        assertTrue(NJ.zeros(DType.FLOAT64, 1 << 18).isNative(), "2 MiB is above the 1 MiB threshold");
    }

    @Test
    void scopeFactoriesAllocateScopedNativeArrays() {
        NDArray kept;
        try (NJScope scope = NJ.scope()) {
            NDArray z = scope.zeros(DType.INT32, 2, 2);
            NDArray o = scope.ones(DType.FLOAT64, 3);
            NDArray e = scope.empty(DType.UINT8, 4);
            NDArray f = scope.full(DType.FLOAT32, 1.5, 2);
            NDArray a = scope.array(new double[]{1, 2, 3, 4}, 2, 2);
            NDArray r = scope.arange(0, 5, 2);
            for (NDArray x : new NDArray[]{z, o, e, f, a, r}) {
                assertTrue(x.isNative());
                assertSame(scope, x.memoryOwner());
            }
            assertEquals(4.0, a.getDouble(1, 1));
            assertEquals(1.5, f.getDouble(1));
            assertEquals(4, r.getLong(2));
            kept = a;
        }
        assertFalse(kept.isAlive());
    }

    @Test
    void factoriesOnAClosedScopeFail() {
        NJScope scope = NJ.scope();
        scope.close();
        assertThrows(MemoryException.class, () -> scope.zeros(DType.FLOAT64, 3));
    }
}
