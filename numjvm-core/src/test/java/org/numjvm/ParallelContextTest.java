package org.numjvm;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParallelContextTest {
    @Test void validatesWorkerLimitAndClosesIdempotently() {
        assertThrows(IllegalArgumentException.class, () -> NJ.parallel(0));
        assertThrows(IllegalArgumentException.class, () -> NJ.parallel(257));
        var context = NJ.parallel(2);
        assertEquals(2, context.workers());
        assertFalse(context.isClosed());
        context.close();
        context.close();
        assertTrue(context.isClosed());
    }
}
