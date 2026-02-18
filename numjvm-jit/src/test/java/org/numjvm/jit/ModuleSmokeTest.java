package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class ModuleSmokeTest {
    @Test
    void coreIsAvailableToJitConsumers() {
        assertNotNull(NJ.version());
    }
}
