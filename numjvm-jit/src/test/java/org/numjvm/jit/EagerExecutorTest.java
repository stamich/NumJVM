package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class EagerExecutorTest {
    @Test void sharedDagWeakScalarsAndMixedTypesMatchEager() {
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var expression = context.trace(2, i -> {
                var square = i.input(0).square();
                return i.constant(2).subtract(square).multiply(square).add(i.input(1));
            });
            NDArray a = NJ.arange(1, 33, 1).astype(DType.FLOAT32);
            NDArray b = NJ.ones(DType.FLOAT64, 32);
            var inputs = new NDArray[]{a, b};
            var plan = new InvocationPlan(expression, inputs);
            NDArray output = plan.allocate();
            assertFalse(EagerExecutor.tryRun(expression, inputs, output, 0));
            assertTrue(EagerExecutor.tryRun(expression, inputs, output, 1_000_000));
            NDArray square = Ufuncs.SQUARE.apply(a);
            NDArray expected = Ufuncs.ADD.apply(Ufuncs.MULTIPLY.apply(Ufuncs.SUBTRACT.apply(2.0, square), square), b);
            for (int i = 0; i < 32; i++) assertEquals(expected.getDouble(i), output.getDouble(i));
        }
    }

    @Test void identicalOutputAndIdentityGraphAreSafe() {
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var input = NJ.ones(DType.FLOAT64, 20);
            var expression = context.trace(1, i -> i.input(0).add(2).square());
            assertTrue(EagerExecutor.tryRun(expression, new NDArray[]{input}, input, 10000));
            assertEquals(9, input.getDouble(19));
            var identity = context.trace(1, i -> i.input(0));
            assertTrue(EagerExecutor.tryRun(identity, new NDArray[]{input}, input, 0));
            assertEquals(9, input.getDouble(0));
        }
    }

    @Test void explicitEagerBudgetRejectsBeforeWriting() {
        try (var context = Jit.context(Jit.Mode.EAGER, 4, 2048, 0)) {
            var input = NJ.ones(DType.FLOAT64, 20);
            var output = NJ.ones(DType.FLOAT64, 20);
            var expression = context.trace(1, i -> i.input(0).square().add(2));
            assertThrows(IllegalStateException.class, () -> expression.evalInto(output, input));
            assertEquals(1, output.getDouble(0));
        }
    }
}
