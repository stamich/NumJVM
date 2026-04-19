package org.numjvm.jit;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import static org.junit.jupiter.api.Assertions.*;

class FusedMixedTest {
    @Test void mixedSpecialValuesAndBroadcastFollowScalarGraph() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        float[] data = new float[35];
        for (int i = 0; i < data.length; i++) data[i] = i - 7;
        data[0] = -0.0f; data[1] = Float.NaN; data[2] = Float.NEGATIVE_INFINITY;
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(2, i -> i.input(0).negative().absolute().sqrt().add(i.input(1)));
            NDArray[] inputs = {NJ.array(data), NJ.array(new double[]{2}).reshapeView().broadcastTo(35)};
            var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT32, DType.FLOAT64});
            assertNotNull(kernel);
            var actual = new InvocationPlan(graph, inputs).allocate();
            assertTrue(GeneratedExecutor.eligible(inputs, actual));
            GeneratedExecutor.run(kernel, inputs, actual);
            var expected = graph.eval(inputs);
            for (int i = 0; i < data.length; i++) {
                double a = actual.getDouble(i), e = expected.getDouble(i);
                if (Double.isNaN(e)) assertTrue(Double.isNaN(a));
                else assertEquals(Double.doubleToRawLongBits(e), Double.doubleToRawLongBits(a));
            }
        }
    }

    @Test void roundsFloatIntermediateBeforePromotion() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(2, i -> i.input(0).add(Math.scalb(1.0, -24)).add(i.input(1)));
            var left = NJ.ones(DType.FLOAT32, 35);
            var right = NJ.multiply(NJ.ones(DType.FLOAT64, 35), Math.scalb(1.0, -25));
            NDArray[] inputs = {left, right};
            var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT32, DType.FLOAT64});
            assertNotNull(kernel);
            var actual = new InvocationPlan(graph, inputs).allocate();
            GeneratedExecutor.run(kernel, inputs, actual);
            var expected = graph.eval(inputs);
            for (int i = 0; i < 35; i++) {
                assertEquals(1 + Math.scalb(1.0, -25), actual.getDouble(i));
                assertEquals(Double.doubleToRawLongBits(expected.getDouble(i)), Double.doubleToRawLongBits(actual.getDouble(i)));
            }
        }
    }
}
