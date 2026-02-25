package org.numjvm.jit;

import java.util.Objects;

/** An immutable symbolic expression node belonging to one trace. */
public final class Expr {
    enum Op { INPUT, CONSTANT, ADD, SUBTRACT, MULTIPLY, DIVIDE, NEGATIVE, ABSOLUTE, SQUARE, SQRT }
    final Jit.Inputs owner;
    final Op op;
    final int slot;
    final long bits;
    final Expr left;
    final Expr right;

    Expr(Jit.Inputs owner, Op op, int slot, long bits, Expr left, Expr right) {
        this.owner = owner; this.op = op; this.slot = slot; this.bits = bits;
        this.left = left; this.right = right;
    }
    private Expr unary(Op operation) {
        owner.checkOpen();
        return owner.node(operation, -1, 0, this, null);
    }
    private Expr binary(Op operation, Expr other) {
        owner.checkOpen();
        if (Objects.requireNonNull(other, "other").owner != owner) throw new IllegalArgumentException("foreign expression");
        return owner.node(operation, -1, 0, this, other);
    }
    private Expr constant(double value) { return owner.constant(value); }
    /** Adds two expressions. */ public Expr add(Expr other) { return binary(Op.ADD, other); }
    /** Adds a weak double. */ public Expr add(double value) { return add(constant(value)); }
    /** Subtracts two expressions. */ public Expr subtract(Expr other) { return binary(Op.SUBTRACT, other); }
    /** Subtracts a weak double. */ public Expr subtract(double value) { return subtract(constant(value)); }
    /** Multiplies two expressions. */ public Expr multiply(Expr other) { return binary(Op.MULTIPLY, other); }
    /** Multiplies by a weak double. */ public Expr multiply(double value) { return multiply(constant(value)); }
    /** Divides by an expression. */ public Expr divide(Expr other) { return binary(Op.DIVIDE, other); }
    /** Divides by a weak double. */ public Expr divide(double value) { return divide(constant(value)); }
    /** Negates the expression. */ public Expr negative() { return unary(Op.NEGATIVE); }
    /** Takes the absolute value. */ public Expr absolute() { return unary(Op.ABSOLUTE); }
    /** Squares the expression. */ public Expr square() { return unary(Op.SQUARE); }
    /** Takes the square root. */ public Expr sqrt() { return unary(Op.SQRT); }
}
