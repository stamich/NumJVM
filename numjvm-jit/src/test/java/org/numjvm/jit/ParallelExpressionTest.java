package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;

class ParallelExpressionTest {
    @Test void evaluatesContiguousExpressionsAndConfinedFallback() {
        try (var pool = NJ.parallel(3); var jit = Jit.context(Jit.Mode.AUTO); var scope = NJ.scope()) {
            var expression = jit.trace(2, i -> i.input(0).add(i.input(1)).square());
            var left = NJ.ones(DType.FLOAT32, 256, 256);
            var right = NJ.ones(DType.FLOAT32, 256, 256);
            var output = NJ.zeros(DType.FLOAT32, 256, 256);
            assertSame(output, expression.evalInto(pool, output, left, right));
            assertEquals("parallel", pool.lastDecision().strategy());
            assertEquals(4.0, output.getDouble(255, 255));
            var confined = scope.ones(DType.FLOAT32, 65536);
            expression.evalInto(pool, confined, confined, confined);
            assertEquals("sequential", pool.lastDecision().strategy());
            assertEquals(4.0, confined.getDouble(65535));
        }
    }
}
