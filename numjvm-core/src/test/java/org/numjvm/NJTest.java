package org.numjvm;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;

class NJTest {

    @Test
    void versionIsAvailable() {
        String v = NJ.version();
        assertNotNull(v);
        assertFalse(v.isBlank());
    }
}
