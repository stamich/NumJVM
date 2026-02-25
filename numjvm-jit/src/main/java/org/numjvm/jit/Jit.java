package org.numjvm.jit;

import java.util.Objects;
import java.util.function.Function;

/** Entry point for explicit symbolic expression tracing. */
public final class Jit {
    private Jit() { }

    /** Creates a context for expression plans. */
    public static Context context() { return new Context(); }

    /** Owns expression tracing and its runtime resources. */
    public static final class Context implements AutoCloseable {
        private boolean closed;
        private Context() { }

        /** Runs the builder once and validates its symbolic output. */
        public Expression trace(int arity, Function<Inputs, Expr> builder) {
            if (closed) throw new IllegalStateException("context is closed");
            if (arity < 1 || arity > 16) throw new IllegalArgumentException("arity must be between 1 and 16");
            Inputs inputs = new Inputs(arity);
            Expr result;
            try {
                result = Objects.requireNonNull(builder, "builder").apply(inputs);
            } finally {
                inputs.closed = true;
            }
            if (result == null || result.owner != inputs) throw new IllegalArgumentException("foreign or null expression");
            // A single input node can only use one input slot until operations are introduced.
            if (arity != 1 || result.slot != 0) throw new IllegalArgumentException("unused input slot");
            return new Expression(arity, result);
        }

        @Override public void close() { closed = true; }
    }

    /** Symbolic inputs available while a trace callback runs. */
    public static final class Inputs {
        private final int arity;
        private boolean closed;
        private Inputs(int arity) { this.arity = arity; }

        /** Returns an input slot. */
        public Expr input(int slot) {
            if (closed) throw new IllegalStateException("trace has ended");
            if (slot < 0 || slot >= arity) throw new IllegalArgumentException("invalid input slot: " + slot);
            return new Expr(this, slot);
        }
    }
}
