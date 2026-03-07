package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class ExpressionEvaluationTest {
    @Test void largeGraphsUseBlocksOrFailBeforeWritingWhenRequired() {
        for (Jit.Mode mode : new Jit.Mode[]{Jit.Mode.AUTO, Jit.Mode.REQUIRED}) {
            try (var context = Jit.context(mode)) {
                var graph = context.trace(1, i -> {
                    Expr expression = i.input(0);
                    for (int n = 0; n < 70; n++) expression = expression.negative();
                    return expression;
                });
                NDArray output = NJ.ones(DType.FLOAT64, 5);
                if (mode == Jit.Mode.REQUIRED) {
                    assertThrows(IllegalArgumentException.class, () -> graph.evalInto(output, output));
                } else {
                    graph.evalInto(output, output);
                }
                assertEquals(1.0, output.getDouble(0));
            }
        }
    }
    @Test void runsAllModesAndProtectsShiftedOutput() {
        for (Jit.Mode mode : Jit.Mode.values()) {
            try (var context = Jit.context(mode)) {
                var graph = context.trace(1, i -> i.input(0).add(2).square());
                NDArray values = NJ.ones(DType.FLOAT64, 5);
                assertEquals(9.0, graph.eval(values).getDouble(4));
                NDArray storage = NJ.arange(0, 6, 1).astype(DType.FLOAT64);
                NDArray input = storage.slice(Slice.range(0, 5));
                NDArray output = storage.slice(Slice.range(1, 6));
                assertSame(output, graph.evalInto(output, input));
                assertEquals(4.0, output.getDouble(0));
                assertEquals(36.0, output.getDouble(4));
            }
        }
    }
    @Test void requiredRejectsUnsupportedLayoutBeforeWriting() {
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(1, i -> i.input(0).add(2));
            NDArray reversed = NJ.ones(DType.FLOAT64, 5).slice(Slice.step(-1));
            NDArray output = NJ.ones(DType.FLOAT64, 5);
            assertThrows(IllegalArgumentException.class, () -> graph.evalInto(output, reversed));
            assertEquals(1.0, output.getDouble(0));
        }
    }
}
