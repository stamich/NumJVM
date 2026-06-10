package org.numjvm;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParallelContextTest {
    @Test void fixedChunksCoverEveryElementOnce() {
        var counts = new java.util.concurrent.atomic.AtomicIntegerArray(103);
        try (var context = NJ.parallel(3)) {
            context.runChunks(new ChunkPlan(103, 16), (chunk, start, end) -> {
                for (long i = start; i < end; i++) counts.incrementAndGet((int) i);
            });
        }
        for (int i = 0; i < 103; i++) assertEquals(1, counts.get(i));
        var plan = new ChunkPlan(Long.MAX_VALUE, 8192);
        assertEquals(Long.MAX_VALUE, plan.end(plan.count - 1));
        assertEquals(0, new ChunkPlan(0, 3).count);
    }

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
