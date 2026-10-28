package org.numjvm.internal;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;

class SharedSecretsTest {

    @Test
    void accessorsAreRegisteredOnFirstUse() {
        assertNotNull(SharedSecrets.arrayAccess());
        assertNotNull(SharedSecrets.scopeAccess());
        assertSame(SharedSecrets.arrayAccess(), SharedSecrets.arrayAccess());
        assertSame(SharedSecrets.scopeAccess(), SharedSecrets.scopeAccess());
    }
}
