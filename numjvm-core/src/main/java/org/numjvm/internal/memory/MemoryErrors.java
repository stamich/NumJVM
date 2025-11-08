package org.numjvm.internal.memory;

import org.numjvm.exception.MemoryException;
import org.numjvm.memory.MemoryOwner;

/**
 * Translates lifecycle exceptions of the Foreign Function &amp; Memory API into
 * {@link MemoryException}s that name the NumJVM owner involved.
 */
public final class MemoryErrors {

    private MemoryErrors() {
    }

    /**
     * Translates an exception raised while accessing memory owned by {@code owner}.
     *
     * @param e     the exception; {@link IllegalStateException} for a closed arena,
     *              {@link WrongThreadException} for access from a thread the arena does not allow
     * @param owner the owner of the memory
     * @return a {@link MemoryException} for FFM lifecycle errors, otherwise {@code e} itself
     */
    public static RuntimeException translate(RuntimeException e, MemoryOwner owner) {
        if (e instanceof WrongThreadException) {
            return new MemoryException(owner.describe() + " is confined to another thread; it cannot be used from "
                    + Thread.currentThread().getName() + " (use NJ.scopeShared() for multi-threaded access)", e);
        }
        if (e instanceof IllegalStateException && !owner.isAlive()) {
            return new MemoryException(owner.describe() + " is closed; its arrays can no longer be accessed", e);
        }
        if (e instanceof IllegalStateException) {
            return new MemoryException("memory of " + owner.describe() + " is not accessible: " + e.getMessage(), e);
        }
        return e;
    }
}
