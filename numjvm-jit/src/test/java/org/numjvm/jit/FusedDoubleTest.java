package org.numjvm.jit;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class FusedDoubleTest {
    @Test void specialValuesAndEveryOperationMatchScalar() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        double[] raw = {-0.0, 0.0, -1, 1, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.MIN_VALUE, -Double.MIN_VALUE, 4, -4, 9, 16, 25, 36, 49, 64, 81, 100};
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            NDArray input = NJ.array(raw);
            for (int op = 0; op < 8; op++) {
                final int selected = op;
                var graph = context.trace(1, i -> {
                    var x = i.input(0);
                    return switch (selected) {
                        case 0 -> x.add(x); case 1 -> x.subtract(x); case 2 -> x.multiply(x);
                        case 3 -> x.divide(x); case 4 -> x.negative(); case 5 -> x.absolute();
                        case 6 -> x.square(); default -> x.sqrt();
                    };
                });
                var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT64});
                assertNotNull(kernel);
                var inputs = new NDArray[]{input};
                var actual = new InvocationPlan(graph, inputs).allocate();
                GeneratedExecutor.run(kernel, inputs, actual);
                var expected = graph.eval(inputs);
                for (int i = 0; i < raw.length; i++) {
                    double a = actual.getDouble(i), e = expected.getDouble(i);
                    if (Double.isNaN(e)) assertTrue(Double.isNaN(a));
                    else assertEquals(Double.doubleToRawLongBits(e), Double.doubleToRawLongBits(a));
                }
            }
        }
    }

    @Test void generatedConstantsAndSharedNodesArePreserved() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(1, i -> {
                var x = i.input(0).add(.25);
                return x.multiply(x).add(x);
            });
            var inputs = new NDArray[]{NJ.ones(35)};
            var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT64});
            assertNotNull(kernel);
            var actual = new InvocationPlan(graph, inputs).allocate();
            GeneratedExecutor.run(kernel, inputs, actual);
            assertEquals(2.8125, actual.getDouble(0));
            assertEquals(2.8125, actual.getDouble(34));
        }
    }

    @Test void generatedDoubleVectorsMatchScalarIncludingTailAndOffsets() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(2, i -> {
                var x = i.input(0); var y = i.input(1);
                return x.multiply(y).add(x).subtract(y).divide(y).negative().absolute().square().sqrt();
            });
            NDArray a = NJ.add(NJ.arange(1, 132, 1).astype(DType.FLOAT64), .25).slice(Slice.range(1, 130));
            NDArray b = NJ.add(NJ.arange(1, 132, 1).astype(DType.FLOAT64), 2).slice(Slice.range(1, 130));
            NDArray[] inputs = {a, b};
            var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT64, DType.FLOAT64});
            assertNotNull(kernel);
            var output = new InvocationPlan(graph, inputs).allocate();
            GeneratedExecutor.run(kernel, inputs, output);
            var scalar = graph.eval(inputs);
            for (int i = 0; i < 129; i++)
                assertEquals(Double.doubleToRawLongBits(scalar.getDouble(i)), Double.doubleToRawLongBits(output.getDouble(i)));
            assertNull(FusedProviders.compile(graph, new DType[]{DType.FLOAT32, DType.FLOAT64}));
        }
    }
}
