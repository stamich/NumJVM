package org.numjvm.memory;

import java.lang.foreign.Arena;
import java.util.concurrent.atomic.AtomicLong;

import org.numjvm.exception.MemoryException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.internal.memory.NativeAllocator;

/**
 * Deterministic lifetime for array memory.
 *
 * <p>Arrays created through a scope live in native memory owned by the scope. Closing
 * the scope frees that memory at once; every array and view created from it then
 * throws {@link MemoryException} on access. Use a scope for temporary arrays in loops:
 * allocation and release are several times cheaper than for GC-managed arrays.
 *
 * <pre>{@code
 * try (NJScope scope = NJ.scope()) {
 *     NDArray a = scope.zeros(DType.FLOAT64, 1000, 1000);
 *     NDArray b = a.transpose();   // a view: same memory, same lifetime
 *     result = scope.keep(b);      // a copy that outlives the scope
 * }
 * }</pre>
 *
 * <h2>Threads</h2>
 * <ul>
 *   <li>{@link Mode#CONFINED} (the default, {@code NJ.scope()}): only the thread that
 *       created the scope may access its arrays or close it.</li>
 *   <li>{@link Mode#SHARED} ({@code NJ.scopeShared()}): any thread may access the arrays.
 *       Closing costs more (about 35 µs, a handshake with all threads). Sharing makes
 *       access <em>possible</em>, not <em>safe</em>: concurrent writes to the same
 *       elements still need external synchronisation.</li>
 * </ul>
 */
public final class NJScope implements MemoryOwner, AutoCloseable {

    /** Which threads may use a scope. */
    public enum Mode {
        /** Only the creating thread. */
        CONFINED,
        /** Any thread. */
        SHARED
    }

    private static final AtomicLong IDS = new AtomicLong();

    static {
        SharedSecrets.setScopeAccess(scope -> scope.allocator);
    }

    private final Arena arena;
    private final Mode mode;
    private final long id;
    private final String ownerThread;
    private final MemoryAllocator allocator;
    private volatile boolean closed;

    private NJScope(Mode mode) {
        this.mode = mode;
        this.arena = mode == Mode.CONFINED ? Arena.ofConfined() : Arena.ofShared();
        this.id = IDS.incrementAndGet();
        this.ownerThread = Thread.currentThread().getName();
        this.allocator = NativeAllocator.inArena(arena, this);
    }

    /**
     * Opens a scope that only the current thread may use. Same as {@code NJ.scope()}.
     *
     * @return a new, open scope
     */
    public static NJScope confined() {
        return new NJScope(Mode.CONFINED);
    }

    /**
     * Opens a scope whose arrays any thread may access. Same as {@code NJ.scopeShared()}.
     *
     * @return a new, open scope
     */
    public static NJScope shared() {
        return new NJScope(Mode.SHARED);
    }

    /** @return whether this scope is confined to one thread or shared */
    public Mode mode() {
        return mode;
    }

    @Override
    public boolean isAlive() {
        return !closed && arena.scope().isAlive();
    }

    @Override
    public String describe() {
        return (mode == Mode.CONFINED ? "confined" : "shared") + " NJScope#" + id
                + (mode == Mode.CONFINED ? " (thread " + ownerThread + ")" : "");
    }

    /**
     * Frees all memory of this scope. Arrays and views created from it can no longer be
     * accessed. Closing an already closed scope does nothing.
     *
     * @throws MemoryException if a confined scope is closed from another thread
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        try {
            arena.close();
            closed = true;
        } catch (WrongThreadException e) {
            throw MemoryErrors.translate(e, this);
        } catch (IllegalStateException e) {
            // Already closed through another path; closing is idempotent.
            closed = true;
        }
    }

    @Override
    public String toString() {
        return describe() + (isAlive() ? "" : " [closed]");
    }
}
