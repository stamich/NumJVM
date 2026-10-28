package org.numjvm.internal;

import org.numjvm.dtype.DType;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;

/**
 * Access to non-public members of public classes in other packages of this module.
 *
 * <p>The public classes register an accessor in their static initialiser; callers
 * obtain it here. This keeps internals out of the public API without merging every
 * class into one package. The same pattern is used inside the JDK.
 */
public final class SharedSecrets {

    /**
     * Internal operations on {@link NJScope}.
     */
    public interface ScopeAccess {

        /**
         * Returns the allocator that takes memory from the scope's arena.
         *
         * @param scope the scope
         * @return its allocator
         */
        MemoryAllocator allocator(NJScope scope);

        /**
         * Copies bytes out of a scope into GC-managed storage that outlives it; the
         * memory-level half of {@code NJScope.keep}.
         *
         * @param scope  the scope that is still open
         * @param source buffer to copy from (normally owned by {@code scope})
         * @param start  first byte to copy
         * @param length number of bytes
         * @return a new automatic buffer holding the copy
         */
        Buffer keep(NJScope scope, Buffer source, long start, long length);

        /**
         * Returns the arena behind a scope, for memory that is not allocated but
         * mapped into it (memory-mapped files).
         *
         * @param scope an open scope
         * @return its arena
         */
        java.lang.foreign.Arena arena(NJScope scope);
    }

    /**
     * Internal operations on {@link NDArray}.
     */
    public interface ArrayAccess {

        /**
         * Creates an array on a buffer, validating all invariants.
         *
         * @param buffer    the memory
         * @param dtype     element type
         * @param offset    byte offset of element {@code [0, ..., 0]}
         * @param shape     the shape
         * @param strides   byte strides
         * @param writeable whether elements may be written
         * @return the array
         */
        NDArray create(Buffer buffer, DType dtype, long offset, Shape shape, Strides strides, boolean writeable);

        /**
         * Allocates a zero-filled contiguous array.
         *
         * @param allocator where the memory comes from
         * @param dtype     element type
         * @param shape     the shape
         * @param order     memory order
         * @return the array
         */
        NDArray allocate(MemoryAllocator allocator, DType dtype, Shape shape, Order order);

        /**
         * Copies an array into new contiguous memory from {@code allocator}, converting its
         * dtype: {@code astype} with a chosen allocator.
         *
         * @param array     the source
         * @param allocator where the copy's memory comes from
         * @param target    dtype of the copy
         * @param order     memory order of the copy
         * @return the copy
         */
        NDArray convert(NDArray array, MemoryAllocator allocator, DType target, Order order);

        /**
         * Returns its buffer.
         *
         * @param array an array
         * @return its buffer
         */
        Buffer buffer(NDArray array);

        /**
         * Returns byte offset of its element {@code [0, ..., 0]}.
         *
         * @param array an array
         * @return byte offset of its element {@code [0, ..., 0]}
         */
        long offset(NDArray array);
    }

    private static volatile ScopeAccess scopeAccess;
    private static volatile ArrayAccess arrayAccess;

    private SharedSecrets() {
    }

    /**
     * Registers the scope accessor; called once by {@link NJScope}.
     *
     * @param access the accessor
     */
    public static void setScopeAccess(ScopeAccess access) {
        scopeAccess = access;
    }

    /**
     * Returns the scope accessor, initialising {@link NJScope} if necessary.
     *
     * @return the scope accessor; never {@code null}
     * @throws IllegalStateException if {@link NJScope} did not register an accessor while
     *                               being initialised (a bug, or a call made from inside its
     *                               static initialiser)
     */
    public static ScopeAccess scopeAccess() {
        ScopeAccess a = scopeAccess;
        if (a == null) {
            ensureInitialized(NJScope.class);
            a = scopeAccess;
            if (a == null) {
                throw notRegistered(NJScope.class);
            }
        }
        return a;
    }

    /**
     * Registers the array accessor; called once by {@link NDArray}.
     *
     * @param access the accessor
     */
    public static void setArrayAccess(ArrayAccess access) {
        arrayAccess = access;
    }

    /**
     * Returns the array accessor, initialising {@link NDArray} if necessary.
     *
     * @return the array accessor; never {@code null}
     * @throws IllegalStateException if {@link NDArray} did not register an accessor while
     *                               being initialised (a bug, or a call made from inside its
     *                               static initialiser)
     */
    public static ArrayAccess arrayAccess() {
        ArrayAccess a = arrayAccess;
        if (a == null) {
            ensureInitialized(NDArray.class);
            a = arrayAccess;
            if (a == null) {
                throw notRegistered(NDArray.class);
            }
        }
        return a;
    }

    private static IllegalStateException notRegistered(Class<?> c) {
        return new IllegalStateException(c.getName() + " did not register its SharedSecrets accessor");
    }

    private static void ensureInitialized(Class<?> c) {
        try {
            Class.forName(c.getName(), true, c.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new AssertionError(e);
        }
    }
}
