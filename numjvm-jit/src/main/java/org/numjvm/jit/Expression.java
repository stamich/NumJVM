package org.numjvm.jit;

import org.numjvm.NJ;
import org.numjvm.Experimental;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;

/** An immutable symbolic expression. */
@Experimental
public final class Expression {
    private final Jit.Context context;
    private final int arity;
    private final Expr root;
    private final GraphPlan graph;
    // Bounded by dtype signatures; values contain graph metadata, never input arrays.
    private final java.util.LinkedHashMap<java.util.List<org.numjvm.dtype.DType>, BlockPlan> blocks =
            new java.util.LinkedHashMap<>(8, .75f, true);

    Expression(Jit.Context context, int arity, Expr root) {
        this.context = context;
        this.arity = arity;
        this.root = root;
        this.graph = new GraphPlan(root);
    }

    /** Returns the number of input slots.
     * @return input count
     */
    public int arity() { return arity; }
    Expr root() { return root; }
    GraphPlan graph() { return graph; }
    int scratchBytes() { return context.scratchBytes(); }

    synchronized BlockPlan blockPlan(org.numjvm.dtype.DType[] types) {
        var key = java.util.List.of(types.clone());
        BlockPlan plan = blocks.get(key);
        if (plan == null) {
            plan = new BlockPlan(graph, types);
            blocks.put(key, plan);
            if (blocks.size() > 8) blocks.remove(blocks.keySet().iterator().next());
        }
        return plan;
    }

    /** Evaluates the expression into a new array.
     * @param inputs invocation inputs
     * @return result array
     */
    public NDArray eval(NDArray... inputs) {
        context.admit();
        var plan = new InvocationPlan(this, inputs);
        var output = plan.allocate();
        execute(inputs, output);
        return output;
    }

    /** Evaluates the expression into an existing array.
     * @param output destination
     * @param inputs invocation inputs
     * @return output
     */
    public NDArray evalInto(NDArray output, NDArray... inputs) {
        context.admit();
        var plan = new InvocationPlan(this, inputs);
        plan.checkOutput(output);
        if (!plan.needsTemporary(output)) {
            execute(inputs, output);
        } else {
            try (var scope = NJ.scope()) {
                var temporary = plan.allocateWith(SharedSecrets.scopeAccess().allocator(scope));
                execute(inputs, temporary);
                plan.copyBack(temporary, output);
            }
        }
        return output;
    }

    private void execute(NDArray[] inputs, NDArray output) {
        if (context.mode() == Jit.Mode.EAGER) {
            if (!EagerExecutor.tryRun(this, inputs, output, context.eagerBytes()))
                throw new IllegalStateException("expression exceeds eager temporary byte budget");
            context.eagerRun();
            context.selected("eager", "explicit mode");
            return;
        }
        if (context.mode() == Jit.Mode.BLOCK) {
            if (output.shape().size() == 0) { context.selected("empty", "zero elements"); return; }
            BlockExecutor.run(this, inputs, output);
            context.blockRun();
            context.selected("block", "explicit mode");
            return;
        }
        boolean eligible = KernelLoader.supports(this) && GeneratedExecutor.eligible(inputs, output);
        if (context.mode() == Jit.Mode.REQUIRED && !eligible)
            throw new IllegalArgumentException("generated execution does not support this layout or graph size");
        long size = output.shape().size();
        if (size == 0) { context.selected("empty", "zero elements"); return; }
        var types = new org.numjvm.dtype.DType[inputs.length];
        for (int i = 0; i < inputs.length; i++) types[i] = inputs[i].dtype();
        boolean cached = eligible && context.cached(this, types);
        boolean homogeneous = true;
        for (int i = 1; i < types.length; i++) homogeneous &= types[i] == types[0];
        boolean vectorProvider = !org.numjvm.BackendInfo.simdProvider().equals("scalar");
        boolean small = size <= 512;
        boolean eagerCandidate = small || (size <= 65536 && homogeneous && vectorProvider && graph.nodes().size() >= 3);
        if (context.mode() == Jit.Mode.AUTO && eagerCandidate && EagerExecutor.tryRun(this, inputs, output, context.eagerBytes())) {
            context.eagerRun();
            context.selected("eager", small ? "small expression" : "SIMD eager candidate");
            return;
        }
        // Provisional thresholds must be calibrated on the reference machine.
        boolean generate = context.mode() == Jit.Mode.REQUIRED
                || (context.mode() == Jit.Mode.AUTO && (cached || size >= 4096));
        if (generate && eligible) {
            GeneratedExecutor.run(context.kernel(this, types), inputs, output);
            context.generatedRun();
            context.selected("generated", context.mode() == Jit.Mode.REQUIRED ? "explicit mode" : cached ? "cached kernel" : "large eligible run");
        } else {
            BlockExecutor.run(this, inputs, output);
            context.blockRun();
            context.selected("block", !eligible ? "unsupported generated layout or graph" : "cold or budget-limited run");
        }
    }
}
