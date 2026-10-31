package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;

class CompilationTimingTest {
    @Test void cacheHitDoesNotAccumulateCompilationTime() {
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var expression = context.trace(1, i -> i.input(0).square());
            var input = NJ.ones(256);
            var output = NJ.zeros(256);
            assertEquals(0, context.compilationNanos());
            expression.evalInto(output, input);
            long compile = context.compilationNanos();
            assertTrue(compile > 0);
            expression.evalInto(output, input);
            assertEquals(compile, context.compilationNanos());
            assertEquals(1, context.statistics().cacheHits());
        }
    }
}
