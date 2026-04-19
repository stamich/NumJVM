package org.numjvm.jit;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class FusedLayoutTest {
    @Test void fortranRunsBroadcastAndIdenticalOutputMatchScalar() {
        Assumptions.assumeTrue(FusedProviders.provider() != null, "optional vector provider unavailable");
        for (DType type : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
            try (var context = Jit.context(Jit.Mode.REQUIRED)) {
                NDArray a = NJ.ones(type, 32, 32).copy(Order.F);
                NDArray row = NJ.arange(1, 33, 1).astype(type).broadcastTo(32, 32);
                var inputs = new NDArray[]{a, row};
                var graph = context.trace(2, i -> i.input(0).add(i.input(1)).square());
                var expected = graph.eval(inputs);
                var kernel = FusedProviders.compile(graph, new DType[]{type, type});
                assertNotNull(kernel);
                assertTrue(GeneratedExecutor.eligible(inputs, a));
                GeneratedExecutor.run(kernel, inputs, a);
                for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++)
                    assertEquals(expected.getDouble(r, c), a.getDouble(r, c));
            }
        }
    }

    @Test void nonunitAndNegativeRunsAreNotEligible() {
        var reversed = NJ.ones(64).slice(Slice.step(-1));
        var strided = NJ.ones(128).slice(Slice.step(2));
        assertFalse(GeneratedExecutor.eligible(new NDArray[]{reversed}, NJ.ones(64)));
        assertFalse(GeneratedExecutor.eligible(new NDArray[]{strided}, NJ.ones(64)));
    }
}
