package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import org.numjvm.internal.memory.Scratch;
import static org.junit.jupiter.api.Assertions.*;

class ScratchBlocksTest {
    @Test void gathersAndScattersReversedRunAndReleasesLease() {
        long before = Scratch.activeBytes();
        var source = MemorySegment.ofArray(new double[]{1, 2, 3});
        var target = MemorySegment.ofArray(new double[3]);
        try (var blocks = new ScratchBlocks(2, 100000)) {
            assertTrue(blocks.blockSize() < 100000);
            var slot = blocks.slot(0);
            ScratchBlocks.gather(source, 16, -8, slot, 8, 3);
            assertEquals(3, slot.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 0));
            ScratchBlocks.scatter(slot, target, 0, 8, 8, 3);
            assertEquals(1, target.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, 16));
            assertThrows(IndexOutOfBoundsException.class, () -> blocks.slot(2));
        }
        assertEquals(before, Scratch.activeBytes());
    }
}
