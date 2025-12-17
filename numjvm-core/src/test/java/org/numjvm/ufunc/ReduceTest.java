package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.exception.ValueException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;

/**
 * Expected values are NumPy 2.5's {@code ufunc.reduce}.
 */
class ReduceTest {

    private static final NDArray M = NJ.arange(0, 24, 1).reshape(2, 3, 4);

    @Test
    void reducesSelectedAxes() {
        assertEquals(276, Ufuncs.ADD.reduce(M).getLong());
        assertEquals(0, Ufuncs.ADD.reduce(M).ndim());
        NDArray a0 = Ufuncs.ADD.reduce(M, 0);
        assertArrayEquals(new long[]{3, 4}, a0.shape().dimensions());
        assertEquals(11 + 23, a0.getLong(2, 3));
        NDArray a12 = Ufuncs.ADD.reduce(M, new int[]{1, -1}, false);
        assertArrayEquals(new long[]{2}, a12.shape().dimensions());
        assertEquals(66, a12.getLong(0));
        assertEquals(210, a12.getLong(1));
        NDArray k = Ufuncs.ADD.reduce(M, new int[]{0, 2}, true);
        assertArrayEquals(new long[]{1, 3, 1}, k.shape().dimensions());
        // M[:, 1, :] holds 4..7 and 16..19
        assertEquals(4 + 5 + 6 + 7 + 16 + 17 + 18 + 19, k.getLong(0, 1, 0));
        // axis=() reduces nothing
        assertTrue(NJ.arrayEquals(M, Ufuncs.ADD.reduce(M, new int[0], false)));
    }

    @Test
    void layoutDoesNotChangeTheResult() {
        NDArray f = M.copy(Order.F);
        NDArray t = M.transpose();
        for (int axis = 0; axis < 3; axis++) {
            assertTrue(NJ.arrayEquals(Ufuncs.ADD.reduce(M, axis), Ufuncs.ADD.reduce(f, axis)));
            assertTrue(NJ.arrayEquals(Ufuncs.MAXIMUM.reduce(M, axis), Ufuncs.MAXIMUM.reduce(f, axis)));
            assertTrue(NJ.arrayEquals(Ufuncs.ADD.reduce(M, axis), Ufuncs.ADD.reduce(t, 2 - axis).transpose()));
        }
    }

    @Test
    void identitiesAndEmptyReductions() {
        NDArray empty = NJ.zeros(0, 3);
        assertTrue(NJ.arrayEquals(NJ.zeros(3), Ufuncs.ADD.reduce(empty, 0)));
        assertTrue(NJ.arrayEquals(NJ.ones(3), Ufuncs.MULTIPLY.reduce(empty, 0)));
        assertEquals(0, Ufuncs.ADD.reduce(empty, 1).size());
        assertEquals(0, Ufuncs.MAXIMUM.reduce(empty, 1).size());
        ValueException e = assertThrows(ValueException.class, () -> Ufuncs.MAXIMUM.reduce(empty, 0));
        assertEquals("zero-size array to reduction operation maximum which has no identity", e.getMessage());
        assertEquals(255, Ufuncs.BITWISE_AND.reduce(NJ.zeros(DType.UINT8, 0)).getLong());
        assertTrue(Ufuncs.LOGICAL_AND.reduce(NJ.zeros(DType.FLOAT64, 0)).getBoolean());
        assertEquals(0L, Ufuncs.ADD.identity().orElseThrow());
        assertTrue(Ufuncs.MAXIMUM.identity().isEmpty());
        // np.sum([-0.0]) == 0.0: the reduction starts from the identity
        assertEquals(0, Double.compare(0.0, Ufuncs.ADD.reduce(NJ.array(new double[]{-0.0})).getDouble()));
        assertEquals(0, Double.compare(-0.0, Ufuncs.MULTIPLY.reduce(NJ.array(new double[]{-0.0})).getDouble()));
    }

    @Test
    void dtypes() {
        assertEquals(DType.INT64, Ufuncs.ADD.reduce(NJ.zeros(DType.INT8, 2)).dtype());
        assertEquals(DType.UINT64, Ufuncs.MULTIPLY.reduce(NJ.zeros(DType.UINT8, 2)).dtype());
        assertEquals(DType.INT64, Ufuncs.ADD.reduce(NJ.array(new boolean[]{true, true})).dtype());
        assertEquals(2, Ufuncs.ADD.reduce(NJ.array(new boolean[]{true, true})).getLong());
        assertEquals(200, Ufuncs.ADD.reduce(NJ.array(DType.INT8, new double[]{100, 100})).getLong());
        assertEquals(DType.INT8, Ufuncs.MAXIMUM.reduce(NJ.zeros(DType.INT8, 2)).dtype());
        assertEquals(DType.FLOAT16, Ufuncs.ADD.reduce(NJ.zeros(DType.FLOAT16, 2)).dtype());
        NDArray l = Ufuncs.LOGICAL_AND.reduce(NJ.array(new double[]{1.5, 0}));
        assertEquals(DType.BOOL, l.dtype());
        assertTrue(!l.getBoolean());
    }

    @Test
    void nanAndOrder() {
        NDArray x = NJ.array(new double[]{1, Double.NaN, 3});
        assertTrue(Double.isNaN(Ufuncs.MAXIMUM.reduce(x).getDouble()));
        assertEquals(3.0, Ufuncs.FMAX.reduce(x).getDouble());
        assertEquals(1.0, Ufuncs.FMIN.reduce(x).getDouble());
        // np.max([0.0, -0.0]) is -0.0: the later of two equal values
        assertEquals(0, Double.compare(-0.0, Ufuncs.MAXIMUM.reduce(NJ.array(new double[]{0.0, -0.0})).getDouble()));
        // non-associative operations reduce one axis in index order: 10 - 1 - 2
        assertEquals(7, Ufuncs.SUBTRACT.reduce(NJ.array(new long[]{10, 1, 2})).getLong());
        assertEquals(7, Ufuncs.SUBTRACT.reduce(NJ.array(new long[]{2, 1, 10}).slice(org.numjvm.ndarray.Slice.step(-1)))
                .getLong());
        ValueException e = assertThrows(ValueException.class, () -> Ufuncs.SUBTRACT.reduce(M));
        assertTrue(e.getMessage().contains("not reorderable"), e.getMessage());
        assertEquals(0 - 3, Ufuncs.SUBTRACT.reduce(NJ.arange(0, 6, 1).reshape(2, 3), 0).getLong(0));
    }

    @Test
    void errors() {
        assertThrows(ShapeException.class, () -> Ufuncs.ADD.reduce(M, new int[]{0, 0}, false));
        assertThrows(ShapeException.class, () -> Ufuncs.ADD.reduce(M, 3));
        assertThrows(DTypeException.class, () -> Ufuncs.LESS.reduce(M));
    }

    @Property(tries = 300)
    void sumOverAnAxisMatchesASlowLoop(@ForAll @IntRange(min = 1, max = 5) int a, @ForAll @IntRange(min = 1, max = 5) int b,
                                       @ForAll @IntRange(min = 0, max = 4) int c, @ForAll @IntRange(min = 0, max = 2) int axis,
                                       @ForAll boolean transposed) {
        NDArray x = NJ.arange(0, (long) a * b * c, 1).reshape(a, b, c);
        if (transposed) {
            x = x.transpose().copy().transpose();
        }
        NDArray r = Ufuncs.ADD.reduce(x, new int[]{axis}, true);
        long[] dims = x.shape().dimensions();
        long[] idx = new long[3];
        for (idx[0] = 0; idx[0] < (axis == 0 ? 1 : dims[0]); idx[0]++) {
            for (idx[1] = 0; idx[1] < (axis == 1 ? 1 : dims[1]); idx[1]++) {
                for (idx[2] = 0; idx[2] < (axis == 2 ? 1 : dims[2]); idx[2]++) {
                    long sum = 0;
                    long[] j = idx.clone();
                    for (j[axis] = 0; j[axis] < dims[axis]; j[axis]++) {
                        sum += x.getLong(j);
                    }
                    assertEquals(sum, r.getLong(idx));
                }
            }
        }
    }

    @Test
    void narrowIntegersWrapAtEveryStep() {
        // np.floor_divide.reduce(np.int8([-128, -1, 5])) == -26: (-128 // -1) wraps to -128 first
        assertEquals(-26, Ufuncs.FLOOR_DIVIDE.reduce(NJ.array(DType.INT8, new double[]{-128, -1, 5})).getLong());
        // np.power.reduce(np.uint8([3, 5, 2])) == (243 ** 2) % 256 == 169
        assertEquals(169, Ufuncs.POWER.reduce(NJ.array(DType.UINT8, new double[]{3, 5, 2})).getLong());
    }
}
