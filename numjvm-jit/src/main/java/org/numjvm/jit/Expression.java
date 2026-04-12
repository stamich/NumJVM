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
        boolean eligible = KernelLoader.supports(this) && GeneratedExecutor.eligible(inputs, output);
        if (context.mode() == Jit.Mode.REQUIRED && !eligible)
            throw new IllegalArgumentException("generated execution does not support this layout or graph size");
        if (output.shape().size() == 0) return;
        // Provisional size threshold; reference-machine benchmarks must tune this policy.
        boolean generate = context.mode() == Jit.Mode.REQUIRED
                || (context.mode() == Jit.Mode.AUTO && output.shape().size() >= 1024);
        if (generate && eligible) {
            var types = new org.numjvm.dtype.DType[inputs.length];
            for (int i = 0; i < inputs.length; i++) types[i] = inputs[i].dtype();
            GeneratedExecutor.run(context.kernel(this, types), inputs, output);
            context.generatedRun();
        } else {
            BlockExecutor.run(this, inputs, output);
            context.blockRun();
        }
    }
}
