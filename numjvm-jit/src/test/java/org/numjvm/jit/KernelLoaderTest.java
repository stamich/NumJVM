package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class KernelLoaderTest {
    @Test void generatesSubtractionDivisionAndUnaryOperations() {
        try (var context = Jit.context()) {
            var graph = context.trace(1, i -> i.constant(5).subtract(i.input(0)).divide(2).negative().absolute());
            var kernel = KernelLoader.create(graph, new DType[]{DType.FLOAT32});
            var input = NJ.array(new float[]{1.2f});
            var eager = Ufuncs.ABSOLUTE.apply(Ufuncs.NEGATIVE.apply(Ufuncs.DIVIDE.apply(
                    Ufuncs.SUBTRACT.apply(5.0, input), 2.0)));
            assertEquals(Double.doubleToRawLongBits(eager.getDouble(0)),
                    Double.doubleToRawLongBits(kernel.evaluate(new double[]{1.2f})));
        }
    }
    @Test void definesAndInvokesHiddenKernel() {
        try (var context = Jit.context()) {
            var graph = context.trace(2, i -> i.input(0).add(0.25).square().multiply(i.input(1)).sqrt());
            var kernel = KernelLoader.create(graph, new DType[]{DType.FLOAT32, DType.FLOAT64});
            assertTrue(kernel.getClass().isHidden());
            assertEquals(1, kernel.version());
            var a = NJ.array(new float[]{1.3f});
            var b = NJ.array(new double[]{3.0});
            double eager = Ufuncs.SQRT.apply(Ufuncs.MULTIPLY.apply(Ufuncs.SQUARE.apply(Ufuncs.ADD.apply(a, .25)), b)).getDouble(0);
            assertEquals(Double.doubleToRawLongBits(eager), Double.doubleToRawLongBits(kernel.evaluate(new double[]{1.3f, 3})));
        }
    }
}
