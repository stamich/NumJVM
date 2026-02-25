package org.numjvm.jit;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TracingTest {
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
            assertEquals(1, calls.get());
            assertThrows(IllegalStateException.class, () -> captured[0].input(0));
            assertThrows(IllegalArgumentException.class, () -> context.trace(1, inputs -> inputs.input(1)));
            assertThrows(IllegalArgumentException.class, () -> context.trace(2, inputs -> inputs.input(0)));
        }
    }
}
