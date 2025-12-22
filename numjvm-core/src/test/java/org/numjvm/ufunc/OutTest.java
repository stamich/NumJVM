package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;

/**
 * Expected values are NumPy 2.5's for the same {@code out=} calls.
 */
class OutTest {

    @Test
    void writesIntoOut() {
        NDArray out = NJ.zeros(2, 3);
        NDArray r = Ufuncs.ADD.apply(NJ.ones(2, 3), NJ.arange(0.0, 3.0, 1.0), out);
        assertSame(out, r);
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{1, 2, 3, 1, 2, 3}, 2, 3), out));
        NDArray t = NJ.zeros(3, 2).transpose();
        Ufuncs.MULTIPLY.apply(NJ.ones(2, 3), 2.0, t);
        assertEquals(2.0, t.getDouble(1, 2));
    }

    @Test
    void inPlaceOperations() {
        NDArray a = NJ.arange(0.0, 4.0, 1.0);
        assertSame(a, a.addInPlace(10.0));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{10, 11, 12, 13}), a));
        a.multiplyInPlace(NJ.array(new double[]{1, 0, 1, 0}));
        a.subtractInPlace(1L);
        a.divideInPlace(2.0);
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{4.5, -0.5, 5.5, -0.5}), a));
        NDArray i = NJ.array(DType.INT8, new double[]{100, 1});
        i.addInPlace(100L);
        assertEquals(-56, i.getLong(0));  // wraps, as np.int8 += 100
        // np.int8 array += 1.5 is a UFuncTypeError: float64 cannot be cast to int8 with same_kind
        assertThrows(DTypeException.class, () -> i.addInPlace(1.5));
        NDArray x = NJ.array(new double[]{-1, 2});
        Ufuncs.NEGATIVE.apply(x, x);
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{1, -2}), x));
    }

    @Test
    void overlappingOutputsSeeTheOriginalValues() {
        // a = np.arange(4.); a[1:] += a[:-1]  ->  [0, 1, 3, 5]
        NDArray a = NJ.arange(0.0, 4.0, 1.0);
        Ufuncs.ADD.apply(a.slice(Slice.from(1)), a.slice(Slice.to(-1)), a.slice(Slice.from(1)));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{0, 1, 3, 5}), a));
        // b = np.arange(4.); b += b[::-1]  ->  [3, 3, 3, 3]
        NDArray b = NJ.arange(0.0, 4.0, 1.0);
        b.addInPlace(b.slice(Slice.step(-1)));
        assertTrue(NJ.arrayEquals(NJ.full(DType.FLOAT64, 3.0, 4), b));
        // c = np.ones((2, 2)); c += c[0]  ->  [[2, 2], [2, 2]]
        NDArray c = NJ.ones(2, 2);
        c.addInPlace(c.slice(Slice.index(0)));
        assertTrue(NJ.arrayEquals(NJ.full(DType.FLOAT64, 2.0, 2, 2), c));
    }

    @Test
    void outDtypeConversion() {
        // np.add(int8, int8, out=int16) runs the int8 loop (wraps), then converts
        NDArray i8 = NJ.array(DType.INT8, new double[]{100});
        NDArray o16 = NJ.zeros(DType.INT16, 1);
        Ufuncs.ADD.apply(i8, i8, o16);
        assertEquals(-56, o16.getLong(0));
        NDArray f = NJ.zeros(DType.FLOAT32, 1);
        Ufuncs.ADD.apply(NJ.array(new double[]{0.1}), NJ.array(new double[]{0.2}), f);
        assertEquals((float) (0.1 + 0.2), (float) f.getDouble(0));
        NDArray b = NJ.zeros(DType.BOOL, 2);
        Ufuncs.LESS.apply(NJ.array(new long[]{1, 3}), 2L, b);
        assertTrue(b.getBoolean(0));
    }

    @Test
    void invalidOutputs() {
        NDArray a = NJ.ones(2, 3);
        assertThrows(ShapeException.class, () -> Ufuncs.ADD.apply(a, a, NJ.zeros(3)));
        assertThrows(ShapeException.class, () -> Ufuncs.ADD.apply(a, a, NJ.zeros(1, 3)));
        assertThrows(ReadOnlyArrayException.class, () -> Ufuncs.ADD.apply(a, a, NJ.zeros(3).broadcastTo(2, 3)));
        DTypeException e = assertThrows(DTypeException.class, () -> Ufuncs.ADD.apply(a, a, NJ.zeros(DType.INT32, 2, 3)));
        assertEquals("cannot cast ufunc 'add' output from float64 to int32 with casting rule 'same_kind'",
                e.getMessage());
        assertThrows(DTypeException.class, () -> Ufuncs.SQRT.apply(a, NJ.zeros(DType.INT64, 2, 3)));
    }

    @Test
    void reductionsIntoOut() {
        NDArray m = NJ.arange(0, 6, 1).reshape(2, 3);
        NDArray out = NJ.zeros(DType.FLOAT64, 3);
        assertSame(out, Reductions.sum(m, new int[]{0}, false, out));
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{3, 5, 7}), out));
        NDArray keep = NJ.zeros(DType.INT64, 2, 1);
        Reductions.max(m, new int[]{1}, true, keep);
        assertEquals(5, keep.getLong(1, 0));
        Ufuncs.MULTIPLY.reduce(m, null, false, NJ.zeros(DType.INT64));
        NDArray mean = NJ.zeros(DType.FLOAT32, 2);
        Reductions.mean(m, new int[]{1}, false, mean);
        assertEquals(4.0, mean.getDouble(1));
        assertThrows(ShapeException.class, () -> Reductions.sum(m, new int[]{0}, false, NJ.zeros(2)));
        assertThrows(DTypeException.class, () -> Reductions.mean(m, null, false, NJ.zeros(DType.INT64)));
        Reductions.prod(m, new int[]{0}, false, NJ.zeros(DType.INT64, 3));
        Reductions.min(m, null, false, NJ.zeros(DType.INT64));
    }
}
