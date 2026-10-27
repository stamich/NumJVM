package org.numjvm.memory;

import org.numjvm.exception.MemoryException;

/**
 * Decides how long array memory stays valid.
 *
 * <p>Every buffer has exactly one owner, and every array and view keeps a strong
 * reference to its buffer, so a view can never shorten the lifetime of the memory it
 * reads. NumJVM has two kinds of owner:
 * <ul>
 *   <li><b>automatic</b>: memory stays valid while any array that uses it is reachable
 *       and is reclaimed by the garbage collector afterwards. Arrays created through
 *       {@link org.numjvm.NJ} factories use it.</li>
 *   <li><b>scoped</b> ({@link NJScope}): memory is released when the scope is closed;
 *       every array and view using it becomes inaccessible at that moment.</li>
 * </ul>
 */
public interface MemoryOwner {

    /**
     * Tells whether memory owned by this owner may still be accessed.
     *
     * @return {@code false} once the owner has released its memory
     */
    boolean isAlive();

    /**
     * Throws if the owner has released its memory.
     *
     * @throws MemoryException if {@link #isAlive()} is {@code false}
     */
    default void checkAlive() {
        if (!isAlive()) {
            throw new MemoryException(describe() + " is closed; its arrays can no longer be accessed");
        }
    }

    /**
     * Returns a short description for error messages, such as {@code "automatic memory"}
     * or {@code "confined NJScope#3"}.
     *
     * @return a human-readable name
     */
    String describe();
}
