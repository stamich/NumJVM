package org.numjvm.jit;

import java.util.Objects;
import java.util.function.Function;

/** Entry point for explicit symbolic expression tracing. */
public final class Jit {
    private Jit() { }

    /** Creates a context for expression plans. */
    public static Context context() { return new Context(Mode.AUTO, 64); }

    /** Execution strategy for expression plans. */
    public enum Mode { AUTO, BLOCK, REQUIRED }

    /** Creates a context using the requested execution strategy. */
    public static Context context(Mode mode) { return new Context(Objects.requireNonNull(mode, "mode"), 64); }

    /** Creates a context with a bounded kernel cache. */
    public static Context context(Mode mode, int cacheEntries) {
        if (cacheEntries < 1 || cacheEntries > 4096) throw new IllegalArgumentException("cache entries must be 1 to 4096");
        return new Context(Objects.requireNonNull(mode, "mode"), cacheEntries);
    }

    /** Owns expression tracing and its runtime resources. */
    public static final class Context implements AutoCloseable {
        private volatile boolean closed;
        private final Mode mode;
        private final int cacheEntries;
        private final java.util.LinkedHashMap<KernelKey, GeneratedKernel> kernels = new java.util.LinkedHashMap<>(16, .75f, true);
        private long compilations;
        private Context(Mode mode, int cacheEntries) { this.mode = mode; this.cacheEntries = cacheEntries; }
        void checkOpen() { if (closed) throw new IllegalStateException("context is closed"); }
        Mode mode() { return mode; }
        int cacheEntries() { return cacheEntries; }
        synchronized long compilations() { return compilations; }

        synchronized GeneratedKernel kernel(Expression expression, org.numjvm.dtype.DType[] types) {
            checkOpen();
            KernelKey key = KernelKey.of(expression, types);
            GeneratedKernel existing = kernels.get(key);
            if (existing != null) return existing;
            GeneratedKernel compiled = KernelLoader.create(expression, types);
            compilations++;
            kernels.put(key, compiled);
            if (kernels.size() > cacheEntries) kernels.remove(kernels.keySet().iterator().next());
            return compiled;
        }

        /** Runs the builder once and validates its symbolic output. */
        public Expression trace(int arity, Function<Inputs, Expr> builder) {
            checkOpen();
            if (arity < 1 || arity > 16) throw new IllegalArgumentException("arity must be between 1 and 16");
            Inputs inputs = new Inputs(arity);
            Expr result;
            try {
                result = Objects.requireNonNull(builder, "builder").apply(inputs);
            } finally {
                inputs.closed = true;
            }
            if (result == null || result.owner != inputs) throw new IllegalArgumentException("foreign or null expression");
            boolean[] used = new boolean[arity];
            var pending = new java.util.ArrayDeque<Expr>();
            var seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<Expr, Boolean>());
            pending.push(result);
            while (!pending.isEmpty()) {
                Expr node = pending.pop();
                if (!seen.add(node)) continue;
                if (node.op == Expr.Op.INPUT) used[node.slot] = true;
                if (node.left != null) pending.push(node.left);
                if (node.right != null) pending.push(node.right);
            }
            for (boolean slotUsed : used) if (!slotUsed) throw new IllegalArgumentException("unused input slot");
            return new Expression(this, arity, result);
        }

        @Override public synchronized void close() { closed = true; kernels.clear(); }
    }

    /** Symbolic inputs available while a trace callback runs. */
    public static final class Inputs {
        private final int arity;
        private boolean closed;
        private int nodes;
        private Inputs(int arity) { this.arity = arity; }

        void checkOpen() { if (closed) throw new IllegalStateException("trace has ended"); }
        Expr node(Expr.Op op, int slot, long bits, Expr left, Expr right) {
            checkOpen();
            if (++nodes > 128) throw new IllegalArgumentException("expression exceeds node limit");
            return new Expr(this, op, slot, bits, left, right);
        }
        /** Creates a weak double constant. */
        public Expr constant(double value) { return node(Expr.Op.CONSTANT, -1, Double.doubleToRawLongBits(value), null, null); }
        /** Returns an input slot. */
        public Expr input(int slot) {
            checkOpen();
            if (slot < 0 || slot >= arity) throw new IllegalArgumentException("invalid input slot: " + slot);
            return node(Expr.Op.INPUT, slot, 0, null, null);
        }
    }
}
