package org.numjvm.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicLong;

import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.internal.Factories;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.internal.memory.NativeAllocator;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Strides;

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
 * <p>{@code keep} always copies: FFM cannot transfer memory from one arena to another,
 * and a zero-copy promise that depends on implementation details would be worse than an
 * honest copy.
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

    /**
     * Which threads may use a scope.
     */
    public enum Mode {
        /**
         * Only the creating thread.
         */
        CONFINED,
        /**
         * Any thread.
         */
        SHARED
    }

    private static final AtomicLong IDS = new AtomicLong();

    static {
        SharedSecrets.setScopeAccess(new SharedSecrets.ScopeAccess() {
            @Override
            public MemoryAllocator allocator(NJScope scope) {
                return scope.allocator;
            }

            @Override
            public Buffer keep(NJScope scope, Buffer source, long start, long length) {
                return scope.keepBytes(source, start, length);
            }

            @Override
            public Arena arena(NJScope scope) {
                scope.checkAlive();
                return scope.arena;
            }
        });
    }

    private final Arena arena;
    private final Mode mode;
    private final long id;
    private final String ownerThread;
    private final NativeAllocator allocator;
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

    /**
     * Returns whether this scope is confined to one thread or shared.
     *
     * @return whether this scope is confined to one thread or shared
     */
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
            allocator.arenaClosed();
        } catch (WrongThreadException e) {
            throw MemoryErrors.translate(e, this);
        } catch (IllegalStateException e) {
            // Already closed through another path; closing is idempotent.
            closed = true;
        }
    }

    // ------------------------------------------------------------------ keep

    /**
     * Copies an array into automatically managed memory so that it outlives this scope.
     *
     * <p>Always a copy (C order), never an alias: FFM cannot move memory between arenas.
     * The result is independent of the scope and writeable.
     *
     * @param array an array, typically created in this scope
     * @return the copy
     * @throws MemoryException if this scope or the array's memory is already closed
     */
    public NDArray keep(NDArray array) {
        checkAlive();
        array.memoryOwner().checkAlive();
        if (array.isCContiguous() && array.size() > 0) {
            SharedSecrets.ArrayAccess access = SharedSecrets.arrayAccess();
            Buffer kept = keepBytes(access.buffer(array), access.offset(array), array.nbytes());
            return access.create(kept, array.dtype(), 0, array.shape(),
                    Strides.cContiguous(array.shape(), array.itemSize()), true);
        }
        return array.copy();
    }

    // ------------------------------------------------------------------ factories

    /**
     * Returns a new scoped array filled with zeros.
     *
     * @param dtype element type
     * @param shape dimensions
     * @return the array, valid until this scope closes
     * @throws MemoryException if the scope is closed
     */
    public NDArray zeros(DType dtype, long... shape) {
        return Factories.zeros(allocator, dtype, shape);
    }

    /**
     * Returns a new scoped array filled with ones.
     *
     * @param dtype element type (real)
     * @param shape dimensions
     * @return the array, valid until this scope closes
     */
    public NDArray ones(DType dtype, long... shape) {
        return Factories.full(allocator, dtype, 1.0, shape);
    }

    /**
     * Same as {@link #zeros(DType, long...)}: new memory is always zero-filled.
     *
     * @param dtype element type
     * @param shape dimensions
     * @return the array, valid until this scope closes
     */
    public NDArray empty(DType dtype, long... shape) {
        return zeros(dtype, shape);
    }

    /**
     * Returns a new scoped array with every element set to {@code value}.
     *
     * @param dtype element type (real)
     * @param value fill value
     * @param shape dimensions
     * @return the array, valid until this scope closes
     */
    public NDArray full(DType dtype, double value, long... shape) {
        return Factories.full(allocator, dtype, value, shape);
    }

    /**
     * Returns a scoped float64 array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array, valid until this scope closes
     */
    public NDArray array(double[] values, long... shape) {
        return Factories.of(allocator, values, shape);
    }

    /**
     * Returns scoped int64 values in {@code [start, stop)}.
     *
     * @param start first value
     * @param stop  end, exclusive
     * @param step  spacing, not zero
     * @return the array, valid until this scope closes
     */
    public NDArray arange(long start, long stop, long step) {
        return Factories.arange(allocator, start, stop, step);
    }

    /**
     * Copies {@code length} bytes from {@code source} into a GC-managed buffer.
     *
     * <p>FFM cannot move a segment from one arena to another, so keeping data beyond a
     * scope always means copying it (ADR-001). The copy is allocated by the default
     * hybrid allocator: on the heap up to the threshold, otherwise in an automatic arena.
     */
    private Buffer keepBytes(Buffer source, long start, long length) {
        checkAlive();
        Buffer target = HybridAllocator.defaultInstance().allocate(length);
        try {
            MemorySegment.copy(source.segment(), start, target.segment(), 0, length);
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, source.owner());
        }
        return target;
    }

    @Override
    public String toString() {
        return describe() + (isAlive() ? "" : " [closed]");
    }
}
