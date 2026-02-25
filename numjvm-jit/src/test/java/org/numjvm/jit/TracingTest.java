package org.numjvm.jit;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TracingTest {
    @Test void buildsSupportedNodesAndRejectsForeignNodes() {
        try (var context = Jit.context()) {
            var result = context.trace(1, i -> i.constant(2).divide(i.input(0)).add(1).subtract(2)
                    .multiply(3).divide(4).negative().absolute().square().sqrt());
            assertEquals(1, result.arity());
            Expr[] foreign = new Expr[1];
            context.trace(1, i -> foreign[0] = i.input(0));
            assertThrows(IllegalArgumentException.class, () -> context.trace(1, i -> i.input(0).add(foreign[0])));
            assertThrows(IllegalArgumentException.class, () -> context.trace(0, i -> i.input(0)));
            assertThrows(IllegalArgumentException.class, () -> context.trace(1, i -> null));
        }
    }
    @Test void tracesOnceAndClosesInputs() {
        AtomicInteger calls = new AtomicInteger();
        Jit.Inputs[] captured = new Jit.Inputs[1];
        try (var context = Jit.context()) {
            Expression expression = context.trace(1, inputs -> {
                calls.incrementAndGet();
                captured[0] = inputs;
                return inputs.input(0);
            });
            assertEquals(1, expression.arity());
            assertEquals(2, context.trace(2, i -> i.constant(1.0).subtract(i.input(0)).multiply(i.input(1))).arity());
            assertEquals(1, calls.get());
            assertThrows(IllegalStateException.class, () -> captured[0].input(0));
            assertThrows(IllegalArgumentException.class, () -> context.trace(1, inputs -> inputs.input(1)));
            assertThrows(IllegalArgumentException.class, () -> context.trace(2, inputs -> inputs.input(0)));
        }
    }
}
