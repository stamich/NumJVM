package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class BlockExecutorTest {
    @Test void mixedPrecisionAndWeakScalarMatchEagerAcrossBlocks() {
        NDArray a = NJ.arange(0, 10000, 1).astype(DType.FLOAT32);
        NDArray b = NJ.ones(DType.FLOAT64, 10000);
        try (var context = Jit.context()) {
            var graph = context.trace(2, i -> i.input(0).add(0.25).square().multiply(i.input(1)).sqrt());
            var plan = new InvocationPlan(graph, new NDArray[]{a, b});
            NDArray result = plan.allocate();
            BlockExecutor.run(graph, new NDArray[]{a, b}, result);
            NDArray eager = Ufuncs.SQRT.apply(Ufuncs.MULTIPLY.apply(Ufuncs.SQUARE.apply(Ufuncs.ADD.apply(a, .25)), b));
            for (int i = 0; i < 10000; i++)
                assertEquals(Double.doubleToRawLongBits(eager.getDouble(i)), Double.doubleToRawLongBits(result.getDouble(i)));
        }
    }
    @Test void permitsIdenticalOutput() {
        NDArray input = NJ.ones(DType.FLOAT64, 9000);
        try (var context = Jit.context()) {
            var graph = context.trace(1, i -> i.input(0).add(2.0).square());
            BlockExecutor.run(graph, new NDArray[]{input}, input);
            assertEquals(9.0, input.getDouble(8999));
        }
    }
    @Test void directLeavesHandleSharedDagAndHazardousOutput() {
        NDArray base = NJ.arange(0, 12, 1).astype(DType.FLOAT64);
        NDArray input = base.slice(Slice.range(0, 10));
        NDArray output = base.slice(Slice.range(1, 11));
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var graph = context.trace(1, i -> {
                var x = i.input(0);
                return x.multiply(x).add(x);
            });
            graph.evalInto(output, input);
            for (int i = 0; i < 10; i++) assertEquals(i * i + i, output.getDouble(i));
        }
    }
    @Test void identityGraphCopiesStridedInput() {
        NDArray input = NJ.arange(0, 20, 1).astype(DType.FLOAT64).slice(Slice.step(2));
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var graph = context.trace(1, i -> i.input(0));
            NDArray output = graph.eval(input);
            for (int i = 0; i < 10; i++) assertEquals(2 * i, output.getDouble(i));
        }
    }
}
