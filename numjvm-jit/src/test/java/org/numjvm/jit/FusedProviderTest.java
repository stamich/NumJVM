package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;

class FusedProviderTest {
    @Test void discoversProviderWhenAvailableAndAlwaysRunsScalarGeneratedCode() {
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            assertEquals(System.getProperty("numjvm.expectFusedProvider", "unavailable"), context.fusedVectorProvider());
            assertEquals(4, context.trace(1, i -> i.input(0).square()).eval(NJ.add(NJ.ones(4), 1)).getDouble(0));
        }
    }
}
