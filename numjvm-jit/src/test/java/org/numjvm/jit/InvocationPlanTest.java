package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class InvocationPlanTest {
    @Test void detectsShiftedOverlapAndAllowsIdenticalView() {
        NDArray storage = NJ.ones(DType.FLOAT64, 6);
        NDArray input = storage.slice(Slice.range(0, 5));
        NDArray shifted = storage.slice(Slice.range(1, 6));
        try (var context = Jit.context()) {
            var graph = context.trace(1, i -> i.input(0).square());
            var plan = new InvocationPlan(graph, new NDArray[]{input});
            assertFalse(plan.needsTemporary(input));
            assertTrue(plan.needsTemporary(shifted));
            assertFalse(plan.needsTemporary(NJ.ones(DType.FLOAT64, 5)));
            NDArray temporary = NJ.zeros(DType.FLOAT64, 5);
            temporary.setDouble(0, -0.0);
            assertSame(shifted, plan.copyBack(temporary, shifted));
            assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(shifted.getDouble(0)));
        }
    }
    @Test void allocatesMatchingEagerOrderAndValidatesOutput() {
        NDArray a = NJ.ones(DType.FLOAT32, 3, 2).transpose();
        try (var context = Jit.context()) {
            var graph = context.trace(1, i -> i.input(0).square());
            var plan = new InvocationPlan(graph, new NDArray[]{a});
            var result = plan.allocate();
            assertArrayEquals(Ufuncs.SQUARE.apply(a).strides().values(), result.strides().values());
            plan.checkOutput(result);
            assertThrows(DTypeException.class, () -> plan.checkOutput(NJ.ones(DType.FLOAT64, 2, 3)));
            assertThrows(ShapeException.class, () -> plan.checkOutput(NJ.ones(DType.FLOAT32, 6)));
        }
    }
    @Test void rejectsClosedScopeEvenForEmptyInput() {
        var scope = NJ.scope();
        var empty = scope.zeros(DType.FLOAT64, 0);
        scope.close();
        try (var context = Jit.context()) {
            var graph = context.trace(1, i -> i.input(0).square());
            assertThrows(MemoryException.class, () -> new InvocationPlan(graph, new NDArray[]{empty}));
        }
    }
}
