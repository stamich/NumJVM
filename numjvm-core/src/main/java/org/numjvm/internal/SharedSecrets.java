package org.numjvm.internal;

import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.memory.NJScope;

/**
 * Access to non-public members of public classes in other packages of this module.
 *
 * <p>The public classes register an accessor in their static initialiser; callers
 * obtain it here. This keeps internals out of the public API without merging every
 * class into one package. The same pattern is used inside the JDK.
 */
public final class SharedSecrets {

    /** Internal operations on {@link NJScope}. */
    public interface ScopeAccess {

        /**
         * Returns the allocator that takes memory from the scope's arena.
         *
         * @param scope the scope
         * @return its allocator
         */
        MemoryAllocator allocator(NJScope scope);
    }

    private static volatile ScopeAccess scopeAccess;

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

    /** @return the scope accessor, initialising {@link NJScope} if necessary */
    public static ScopeAccess scopeAccess() {
        ScopeAccess a = scopeAccess;
        if (a == null) {
            ensureInitialized(NJScope.class);
            a = scopeAccess;
        }
        return a;
    }

    private static void ensureInitialized(Class<?> c) {
        try {
            Class.forName(c.getName(), true, c.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new AssertionError(e);
        }
    }
}
