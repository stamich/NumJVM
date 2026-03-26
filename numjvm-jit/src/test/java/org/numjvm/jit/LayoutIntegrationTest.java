package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class LayoutIntegrationTest {
    @Test void evaluatesBroadcastWithReversedInputAndSteppedOutput() {
        NDArray input = NJ.arange(0, 12, 1).astype(DType.FLOAT64).reshape(3, 4).slice(Slice.all(), Slice.step(-1));
        NDArray row = NJ.array(new double[]{1, 2, 3, 4});
        NDArray storage = NJ.zeros(DType.FLOAT64, 3, 8);
        NDArray output = storage.slice(Slice.all(), Slice.step(2));
        NDArray expected = Ufuncs.SQUARE.apply(Ufuncs.ADD.apply(input, row));
        for (var mode : new Jit.Mode[]{Jit.Mode.BLOCK, Jit.Mode.AUTO}) {
            try (var context = Jit.context(mode)) {
                var graph = context.trace(2, i -> i.input(0).add(i.input(1)).square());
                graph.evalInto(output, input, row);
                for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++)
                    assertEquals(expected.getDouble(r, c), output.getDouble(r, c));
            }
        }
    }
    @Test void preservesFortranResultAndRejectsReadOnlyOutput() {
        NDArray input = NJ.ones(DType.FLOAT32, 4, 3).transpose();
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            var graph = context.trace(1, i -> i.input(0).square());
            NDArray actual = graph.eval(input);
            assertArrayEquals(Ufuncs.SQUARE.apply(input).strides().values(), actual.strides().values());
            NDArray readonly = NJ.ones(DType.FLOAT32, 1, 4).broadcastTo(3, 4);
            assertThrows(ReadOnlyArrayException.class, () -> graph.evalInto(readonly, input));
        }
    }
}
