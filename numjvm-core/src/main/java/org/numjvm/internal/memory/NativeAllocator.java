package org.numjvm.internal.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import org.numjvm.exception.MemoryException;
import org.numjvm.memory.MemoryOwner;

/**
 * Allocates off-heap buffers aligned to {@link #ALIGNMENT} bytes.
 *
 * <p>Either every allocation gets its own automatic arena (memory is freed by the GC
 * once unreachable) or all allocations come from one arena whose closing frees them
 * together (used by {@code NJScope}).
 *
 * <p>Native memory is reserved against the JDK's direct-memory limit
 * ({@code -XX:MaxDirectMemorySize}, default: the maximum heap size). Exceeding it is
 * reported as a {@link MemoryException}. Every allocation is counted by a
 * {@link NativeMemoryTracker}.
 */
public final class NativeAllocator implements MemoryAllocator {

    /** Alignment of byte 0 of every native buffer, chosen for SIMD and native libraries. */
    public static final int ALIGNMENT = 64;

    private static final NativeAllocator AUTOMATIC =
            new NativeAllocator(null, AutomaticOwner.INSTANCE, NativeMemoryTracker.GLOBAL);

    private final Arena arena;
    private final MemoryOwner owner;
    private final NativeMemoryTracker tracker;
    private final AtomicLong arenaBytes = new AtomicLong();

    private NativeAllocator(Arena arena, MemoryOwner owner, NativeMemoryTracker tracker) {
        this.arena = arena;
        this.owner = owner;
        this.tracker = tracker;
    }

    /**
     * Returns the allocator that gives every buffer its own automatic arena.
     *
     * @return the shared automatic allocator
     */
    public static NativeAllocator automatic() {
        return AUTOMATIC;
    }

    /**
     * Returns an allocator that takes all memory from {@code arena}.
     *
     * @param arena the arena; closing it frees every buffer allocated here
     * @param owner the owner reported by the buffers
     * @return the allocator
     */
    public static NativeAllocator inArena(Arena arena, MemoryOwner owner) {
        return inArena(arena, owner, NativeMemoryTracker.GLOBAL);
    }

    static NativeAllocator inArena(Arena arena, MemoryOwner owner, NativeMemoryTracker tracker) {
        return new NativeAllocator(Objects.requireNonNull(arena, "arena"), Objects.requireNonNull(owner, "owner"),
                tracker);
    }

    static NativeAllocator automatic(NativeMemoryTracker tracker) {
        return new NativeAllocator(null, AutomaticOwner.INSTANCE, tracker);
    }

    /**
     * Records that the arena of this allocator was closed: everything it allocated is
     * released at once. Called by {@code NJScope.close()}.
     */
    public void arenaClosed() {
        tracker.released(arenaBytes.getAndSet(0));
    }

    @Override
    public Buffer allocate(long byteSize) {
        if (byteSize < 0) {
            throw new MemoryException("negative allocation size " + byteSize);
        }
        Arena a = arena != null ? arena : Arena.ofAuto();
        MemorySegment segment;
        try {
            segment = a.allocate(byteSize, ALIGNMENT);
        } catch (OutOfMemoryError e) {
            throw new MemoryException("cannot allocate " + byteSize + " bytes of native memory ("
                    + e.getMessage() + "); raise -XX:MaxDirectMemorySize or use smaller arrays", e);
        } catch (IllegalStateException | WrongThreadException e) {
            throw new MemoryException("cannot allocate from " + owner.describe() + ": " + e.getMessage(), e);
        }
        Buffer buffer = Buffer.of(segment, owner, ALIGNMENT);
        if (byteSize > 0) {
            tracker.allocated(byteSize);
            if (arena == null) {
                tracker.releaseWhenUnreachable(buffer, byteSize);
            } else {
                arenaBytes.addAndGet(byteSize);
            }
        }
        return buffer;
    }

    @Override
    public String toString() {
        return arena == null ? "NativeAllocator[automatic]" : "NativeAllocator[" + owner.describe() + "]";
    }
}
