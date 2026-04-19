package org.numjvm.jit;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import static org.junit.jupiter.api.Assertions.*;

class FusedFloatTest {
    @Test void allOperationsMatchScalar() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        var input = NJ.array(new float[]{-0.0f, 0, -1, 1, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -Float.MIN_VALUE, 4, 9, 16, 25, 36, 49, 64, 81, 100, 3});
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
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
                var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT32});
                assertNotNull(kernel);
                var inputs = new NDArray[]{input};
                var actual = new InvocationPlan(graph, inputs).allocate();
                GeneratedExecutor.run(kernel, inputs, actual);
                var expected = graph.eval(inputs);
                for (int i = 0; i < 19; i++) {
                    float a = (float) actual.getDouble(i), e = (float) expected.getDouble(i);
                    if (Float.isNaN(e)) assertTrue(Float.isNaN(a));
                    else assertEquals(Float.floatToRawIntBits(e), Float.floatToRawIntBits(a));
                }
            }
        }
    }

    @Test void roundsEveryNodeAndMatchesSpecialValuesAtVectorTails() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        float[] raw = new float[35];
        for (int i = 0; i < raw.length; i++) raw[i] = 1.0f + i / 64.0f;
        raw[0] = -0.0f; raw[1] = 0.0f; raw[2] = Float.NaN;
        raw[3] = Float.POSITIVE_INFINITY; raw[4] = Float.NEGATIVE_INFINITY;
        raw[5] = Float.MIN_VALUE; raw[34] = 1.0f;
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(1, i -> i.input(0).add(Math.scalb(1.0, -24)).square().subtract(1).absolute().sqrt());
            NDArray[] inputs = {NJ.array(raw)};
            var kernel = FusedProviders.compile(graph, new DType[]{DType.FLOAT32});
            assertNotNull(kernel);
            var output = new InvocationPlan(graph, inputs).allocate();
            GeneratedExecutor.run(kernel, inputs, output);
            var expected = graph.eval(inputs);
            for (int i = 0; i < raw.length; i++) {
                float e = (float) expected.getDouble(i), a = (float) output.getDouble(i);
                if (Float.isNaN(e)) assertTrue(Float.isNaN(a));
                else assertEquals(Float.floatToRawIntBits(e), Float.floatToRawIntBits(a), "index " + i);
            }
        }
    }
}
