package org.numjvm.jit;

/** A symbolic input belonging to one expression trace. */
public final class Expr {
    final Object owner;
    final int slot;

    Expr(Object owner, int slot) {
        this.owner = owner;
        this.slot = slot;
    }
}
