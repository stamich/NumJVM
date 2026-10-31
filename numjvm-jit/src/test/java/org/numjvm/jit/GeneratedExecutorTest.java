package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class GeneratedExecutorTest {
    @Test void handlesScalarBroadcastOffsetAndInPlaceOutput() {
        NDArray storage = NJ.ones(DType.FLOAT32, 10);
        NDArray a = storage.slice(Slice.range(1, 9));
        NDArray scalar = NJ.array(new float[]{2}).reshape();
        try (var context = Jit.context()) {
            var graph = context.trace(2, i -> i.input(0).add(i.input(1)).square());
            var kernel = KernelLoader.create(graph, new DType[]{DType.FLOAT32, DType.FLOAT32});
            assertTrue(GeneratedExecutor.eligible(new NDArray[]{a, scalar}, a));
            GeneratedExecutor.run(kernel, new NDArray[]{a, scalar}, a);
            assertEquals(9.0, a.getDouble(7));
            assertEquals(1.0, storage.getDouble(0));
        }
    }
    @Test void rejectsNegativeStrideForGeneratedRuns() {
        NDArray a = NJ.ones(DType.FLOAT64, 4).slice(Slice.step(-1));
        assertFalse(GeneratedExecutor.eligible(new NDArray[]{a}, NJ.ones(DType.FLOAT64, 4)));
    }
    @Test void hoistsAllSlotsForMaximumArityAndNonzeroOffsets() {
        NDArray[] inputs = new NDArray[16];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = NJ.ones(DType.FLOAT64, 259).slice(Slice.range(1, 258));
        }
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(16, i -> {
                Expr sum = i.input(0);
                for (int n = 1; n < 16; n++) sum = sum.add(i.input(n));
                return sum;
            });
            var output = graph.eval(inputs);
            assertEquals(16, output.getDouble(0));
            assertEquals(16, output.getDouble(256));
        }
    }
}
