package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KernelLoaderTest {
    @Test void definesAndInvokesHiddenKernel() {
        var kernel = KernelLoader.create();
        assertTrue(kernel.getClass().isHidden());
        assertEquals(1, kernel.version());
    }
}
