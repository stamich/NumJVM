package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import org.numjvm.internal.memory.Scratch;

/** Operation-local bounded storage for temporary block values. */
final class ScratchBlocks implements AutoCloseable {
    private final Scratch lease;
    private final int blockSize;
    private final int slots;

    ScratchBlocks(int slots, int maxElements) {
        if (slots < 1 || slots > 130 || maxElements < 1) throw new IllegalArgumentException("invalid scratch request");
        this.slots = slots;
        // Every slot holds up to eight bytes per element; each slot stays aligned.
        blockSize = Math.min(maxElements, Scratch.CHUNK_BYTES / (slots * 8));
        lease = Scratch.lease(Math.multiplyExact(Math.multiplyExact(slots, blockSize), 8));
    }

    int blockSize() { return blockSize; }
    MemorySegment slot(int index) {
        if (index < 0 || index >= slots) throw new IndexOutOfBoundsException(index);
        return lease.memory().asSlice((long) index * blockSize * 8, (long) blockSize * 8);
    }

    static void gather(MemorySegment source, long offset, long stride, MemorySegment target, int itemSize, int count) {
        if (itemSize <= 0 || count < 0) throw new IllegalArgumentException("invalid copy dimensions");
        if (stride == itemSize) {
            MemorySegment.copy(source, offset, target, 0, (long) count * itemSize);
            return;
        }
        for (int i = 0; i < count; i++) MemorySegment.copy(source, offset + i * stride, target, (long) i * itemSize, itemSize);
    }

    static void scatter(MemorySegment source, MemorySegment target, long offset, long stride, int itemSize, int count) {
        if (itemSize <= 0 || count < 0) throw new IllegalArgumentException("invalid copy dimensions");
        if (stride == itemSize) {
            MemorySegment.copy(source, 0, target, offset, (long) count * itemSize);
            return;
        }
        for (int i = 0; i < count; i++) MemorySegment.copy(source, (long) i * itemSize, target, offset + i * stride, itemSize);
    }

    @Override public void close() { lease.close(); }
}
