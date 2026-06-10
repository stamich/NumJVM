package org.numjvm;

/** Fixed logical chunks, independent of worker count and scheduling. */
final class ChunkPlan {
    final long size, chunkSize, count;
    ChunkPlan(long size, long chunkSize) {
        if (size < 0 || chunkSize <= 0) throw new IllegalArgumentException("invalid chunk dimensions");
        this.size = size;
        this.chunkSize = chunkSize;
        this.count = size == 0 ? 0 : 1 + (size - 1) / chunkSize;
    }
    long start(long index) {
        if (index < 0 || index >= count) throw new IndexOutOfBoundsException();
        return index * chunkSize;
    }
    long end(long index) {
        long start = start(index);
        return start + Math.min(chunkSize, size - start);
    }
}
