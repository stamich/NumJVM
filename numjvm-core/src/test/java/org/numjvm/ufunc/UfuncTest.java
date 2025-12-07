package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.exception.ValueException;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;

class UfuncTest {

    @Test
    void describesItself() {
        assertEquals("add", Ufuncs.ADD.name());
        assertEquals(2, Ufuncs.ADD.nin());
        assertEquals("<ufunc 'add'>", Ufuncs.ADD.toString());
        List<String> types = Ufuncs.ADD.types();
        assertEquals("bool,bool->bool", types.getFirst());
        assertTrue(types.contains("float64,float64->float64"));
    }

    @Test
    void promotesAndConvertsOperands() {
        NDArray r = NJ.add(NJ.array(DType.INT8, new double[]{1, 2}), NJ.array(DType.UINT8, new double[]{200, 255}));
        assertEquals(DType.INT16, r.dtype());
        assertEquals(201, r.getLong(0));
        assertEquals(257, r.getLong(1));
        NDArray f = NJ.add(NJ.array(DType.INT16, new double[]{1}), NJ.array(DType.FLOAT16, new double[]{0.5}));
        assertEquals(DType.FLOAT32, f.dtype());
        assertEquals(1.5, f.getDouble(0));
        assertEquals(DType.FLOAT64, Ufuncs.ADD.resultType(DType.INT64, DType.UINT64));
    }

    @Test
    void boolAddIsLogicalOr() {
        NDArray r = NJ.add(NJ.array(new boolean[]{true, false, false}), NJ.array(new boolean[]{true, true, false}));
        assertEquals(DType.BOOL, r.dtype());
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, true, false}), r));
    }

    @Test
    void weakScalarsKeepTheArrayDtype() {
        NDArray u8 = NJ.array(DType.UINT8, new double[]{1, 250});
        NDArray r = NJ.add(u8, 10L);
        assertEquals(DType.UINT8, r.dtype());
        assertEquals(4, r.getLong(1));  // wraps
        assertEquals(DType.FLOAT64, NJ.add(NJ.array(DType.INT8, new double[]{1}), 1.5).dtype());
        assertEquals(DType.FLOAT32, NJ.add(NJ.array(new float[]{1}), 1.5).dtype());
        assertEquals(DType.INT64, NJ.add(NJ.array(new boolean[]{true}), 1L).dtype());
        assertEquals(DType.FLOAT16, Ufuncs.ADD.apply(100000L, NJ.array(DType.FLOAT16, new double[]{1})).dtype());
        assertEquals(DType.FLOAT64, Ufuncs.ADD.apply(2.0, NJ.array(DType.INT32, new double[]{1})).dtype());
        assertEquals(2.0, NJ.arange(3).add(1.0).getDouble(1));
        assertEquals(2, NJ.arange(3).add(1L).getLong(1));
    }

    @Test
    void outOfRangeScalarsAreValueExceptions() {
        NDArray u8 = NJ.array(DType.UINT8, new double[]{1});
        ValueException e = assertThrows(ValueException.class, () -> NJ.add(u8, 300L));
        assertEquals("Java integer 300 is out of bounds for uint8", e.getMessage());
        assertThrows(ValueException.class, () -> NJ.add(u8, -1L));
        assertThrows(ValueException.class, () -> NJ.add(NJ.array(DType.INT8, new double[]{1}), 128L));
        assertEquals(DType.INT8, NJ.add(NJ.array(DType.INT8, new double[]{1}), -128L).dtype());
        assertThrows(ValueException.class, () -> NJ.add(NJ.zeros(DType.UINT64, 1), -1L));
    }

    @Test
    void broadcastsAndRejectsIncompatibleShapes() {
        NDArray r = NJ.add(NJ.arange(3).reshape(3, 1), NJ.arange(4));
        assertArrayEquals(new long[]{3, 4}, r.shape().dimensions());
        assertEquals(5, r.getLong(2, 3));
        assertThrows(ShapeException.class, () -> NJ.add(NJ.zeros(3), NJ.zeros(4)));
        assertEquals(0, NJ.add(NJ.zeros(0, 3), NJ.zeros(3)).size());
        NDArray s = NJ.add(NJ.scalar(1.0), NJ.scalar(2.0));
        assertEquals(0, s.ndim());
        assertEquals(3.0, s.getDouble());
    }

    @Test
    void complexHasNoLoopYet() {
        DTypeException e = assertThrows(DTypeException.class,
                () -> NJ.add(NJ.zeros(DType.COMPLEX64, 2), NJ.zeros(DType.COMPLEX64, 2)));
        assertTrue(e.getMessage().contains("ufunc 'add' has no loop for (complex64, complex64)"), e.getMessage());
    }

    @Test
    void outputFollowsTheInputsMemoryOrder() {
        NDArray a = NJ.arange(0.0, 12.0, 1.0).reshape(4, 3).transpose();  // F-ordered view
        NDArray r = NJ.add(a, a);
        assertTrue(r.isFContiguous());
        assertTrue(NJ.arrayEquals(NJ.add(a.copy(), a.copy()), r));
        // C wins when operands disagree
        NDArray mixed = NJ.add(a, a.copy());
        assertTrue(mixed.isCContiguous());
        // broadcast operands do not vote
        NDArray b = NJ.add(a, NJ.arange(0.0, 4.0, 1.0).reshape(1, 4).broadcastTo(3, 4));
        assertTrue(b.isFContiguous());
        NDArray f = NJ.zeros(DType.FLOAT64, 2, 3, 4).copy(Order.F);
        assertTrue(NJ.add(f, f).isFContiguous());
    }

    @Test
    void memoryOrderOfAPlan() {
        assertArrayEquals(new int[]{0, 1, 2}, IterationPlan.memoryOrder(new long[]{2, 3, 4}, new long[][]{{96, 32, 8}}));
        assertArrayEquals(new int[]{2, 1, 0}, IterationPlan.memoryOrder(new long[]{2, 3, 4}, new long[][]{{8, 16, 48}}));
        assertArrayEquals(new int[]{1, 0}, IterationPlan.memoryOrder(new long[]{3, 4}, new long[][]{{8, 24}, {0, 0}}));
        assertArrayEquals(new int[]{0, 1}, IterationPlan.memoryOrder(new long[]{3, 4}, new long[][]{{8, 24}, {32, 8}}));
        assertArrayEquals(new int[0], IterationPlan.memoryOrder(new long[0], new long[][]{{}}));
    }

    @Test
    void resolutionIsCached() {
        BinaryUfunc.Resolution r1 = Ufuncs.ADD.resolve(DType.INT32, DType.INT32, DType.INT32);
        BinaryUfunc.Resolution r2 = Ufuncs.ADD.resolve(DType.INT32, DType.INT8, DType.INT32);
        assertSame(r1, r2);
    }
}
