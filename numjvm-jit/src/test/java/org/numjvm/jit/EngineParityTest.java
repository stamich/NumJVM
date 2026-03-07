package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class EngineParityTest {
    @Test void enginesMatchEagerForExceptionalAndFiniteInputs() {
        float[] values = {-Float.MAX_VALUE, -2f, -0.0f, 0f, Float.MIN_VALUE, 0.25f, 3f,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        NDArray input = NJ.array(values);
        NDArray eager = Ufuncs.SQRT.apply(Ufuncs.ABSOLUTE.apply(Ufuncs.SUBTRACT.apply(
                Ufuncs.DIVIDE.apply(Ufuncs.MULTIPLY.apply(input, input), 3.0), 1.0)));
        for (Jit.Mode mode : Jit.Mode.values()) {
            try (var context = Jit.context(mode)) {
                var expression = context.trace(1, i -> {
                    var x = i.input(0);
                    return x.multiply(x).divide(3).subtract(1).absolute().sqrt();
                });
                NDArray actual = expression.eval(input);
                for (int n = 0; n < values.length; n++) {
                    float expected = (float) eager.getDouble(n);
                    float found = (float) actual.getDouble(n);
                    if (Float.isNaN(expected)) assertTrue(Float.isNaN(found));
                    else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(found));
                }
            }
        }
    }
}
