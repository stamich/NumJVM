package org.numjvm.jit;

import java.util.Random;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class RandomExpressionTest {
    @Test void randomChainsMatchEagerAcrossModes() {
        Random random = new Random(90123);
        for (int trial = 0; trial < 50; trial++) {
            float[] values = new float[19];
            for (int i = 0; i < values.length; i++) values[i] = (random.nextFloat() - .5f) * 100;
            values[0] = -0.0f;
            values[1] = Float.NaN;
            NDArray input = NJ.array(values);
            int[] operations = new int[8];
            for (int i = 0; i < operations.length; i++) operations[i] = random.nextInt(7);
            NDArray expected = input;
            for (int op : operations) expected = eager(expected, op);
            for (var mode : Jit.Mode.values()) {
                try (var context = Jit.context(mode)) {
                    var graph = context.trace(1, i -> {
                        Expr value = i.input(0);
                        for (int op : operations) value = symbolic(value, op);
                        return value;
                    });
                    NDArray actual = graph.eval(input);
                    assertEquals(DType.FLOAT32, actual.dtype());
                    for (int i = 0; i < values.length; i++) {
                        float a = (float) actual.getDouble(i), e = (float) expected.getDouble(i);
                        if (Float.isNaN(e)) assertTrue(Float.isNaN(a), "trial " + trial);
                        else assertEquals(Float.floatToRawIntBits(e), Float.floatToRawIntBits(a), "trial " + trial);
                    }
                }
            }
        }
    }

    private static Expr symbolic(Expr x, int op) {
        return switch (op) {
            case 0 -> x.add(.5);
            case 1 -> x.subtract(3);
            case 2 -> x.multiply(1.25);
            case 3 -> x.divide(2);
            case 4 -> x.negative();
            case 5 -> x.absolute();
            default -> x.square();
        };
    }
    private static NDArray eager(NDArray x, int op) {
        return switch (op) {
            case 0 -> Ufuncs.ADD.apply(x, .5);
            case 1 -> Ufuncs.SUBTRACT.apply(x, 3.0);
            case 2 -> Ufuncs.MULTIPLY.apply(x, 1.25);
            case 3 -> Ufuncs.DIVIDE.apply(x, 2.0);
            case 4 -> Ufuncs.NEGATIVE.apply(x);
            case 5 -> Ufuncs.ABSOLUTE.apply(x);
            default -> Ufuncs.SQUARE.apply(x);
        };
    }
}
