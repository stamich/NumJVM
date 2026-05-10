package org.numjvm.jit;

import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.ndarray.NDArray;
import static org.junit.jupiter.api.Assertions.*;

class FusedPropertyStressTest {
    @Test void mixedRandomGraphsMatchScalarAtEveryElement() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        Random random = new Random(20260510);
        float[] raw = new float[4099]; double[] other = new double[4099];
        for (int i = 0; i < raw.length; i++) {
            raw[i] = (random.nextFloat() - .5f) * 10;
            other[i] = .5 + random.nextDouble();
        }
        raw[0] = -0.0f; raw[1] = Float.NaN; raw[2] = Float.POSITIVE_INFINITY;
        raw[3] = Float.NEGATIVE_INFINITY; raw[4] = Float.MIN_VALUE;
        NDArray[] inputs = {NJ.array(raw), NJ.array(other)};
        for (int trial = 0; trial < 20; trial++) {
            int[] operations = new int[8];
            for (int i = 0; i < operations.length; i++) operations[i] = random.nextInt(8);
            try (var scalar = Jit.context(Jit.Mode.REQUIRED); var auto = Jit.context(Jit.Mode.AUTO)) {
                NDArray expected = graph(scalar, operations).eval(inputs);
                NDArray actual = graph(auto, operations).eval(inputs);
                assertEquals("fused-vector", auto.lastDecision().strategy(), "trial " + trial);
                for (int i = 0; i < raw.length; i++) {
                    double e = expected.getDouble(i), a = actual.getDouble(i);
                    if (Double.isNaN(e)) assertTrue(Double.isNaN(a), "trial " + trial + ", index " + i);
                    else assertEquals(Double.doubleToRawLongBits(e), Double.doubleToRawLongBits(a), "trial " + trial + ", index " + i);
                }
            }
        }
    }
    private static Expression graph(Jit.Context context, int[] operations) {
        return context.trace(2, inputs -> {
            Expr value = inputs.input(0);
            for (int op : operations) value = switch (op) {
                case 0 -> value.add(.25); case 1 -> value.subtract(2);
                case 2 -> value.multiply(1.25); case 3 -> value.divide(2);
                case 4 -> value.negative(); case 5 -> value.absolute();
                case 6 -> value.square(); default -> value.absolute().sqrt();
            };
            return value.add(inputs.input(1));
        });
    }
}
