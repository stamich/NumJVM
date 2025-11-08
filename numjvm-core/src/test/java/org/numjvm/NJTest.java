package org.numjvm;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;
import org.numjvm.memory.NJScope;

class NJTest {

    @Test
    void versionIsAvailable() {
        String v = NJ.version();
        assertNotNull(v);
        assertFalse(v.isBlank());
    }

    @Test
    void scopeFactories() {
        try (NJScope a = NJ.scope(); NJScope b = NJ.scopeConfined(); NJScope c = NJ.scopeShared()) {
            org.junit.jupiter.api.Assertions.assertEquals(NJScope.Mode.CONFINED, a.mode());
            org.junit.jupiter.api.Assertions.assertEquals(NJScope.Mode.CONFINED, b.mode());
            org.junit.jupiter.api.Assertions.assertEquals(NJScope.Mode.SHARED, c.mode());
        }
    }
}
