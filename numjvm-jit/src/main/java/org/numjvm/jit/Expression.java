package org.numjvm.jit;

/** An immutable symbolic expression. */
public final class Expression {
    private final int arity;
    private final Expr root;

    Expression(int arity, Expr root) {
        this.arity = arity;
        this.root = root;
    }

    /** Returns the number of input slots. */
    public int arity() { return arity; }

    Expr root() { return root; }
}
