package org.numjvm.internal.memory;

import org.numjvm.memory.MemoryOwner;

/**
 * Owner of GC-managed memory: heap arrays and automatic arenas. Always alive; memory is
 * reclaimed when nothing references it any more.
 */
public final class AutomaticOwner implements MemoryOwner {

    /**
     * The single instance.
     */
    public static final AutomaticOwner INSTANCE = new AutomaticOwner();

    private AutomaticOwner() {
    }

    @Override
    public boolean isAlive() {
        return true;
    }

    @Override
    public String describe() {
        return "automatic memory";
    }

    @Override
    public String toString() {
        return describe();
    }
}
