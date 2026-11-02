package org.numjvm.jit;

import org.numjvm.NJ;
import org.numjvm.ParallelContext;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
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

    /** Evaluates into an output using an opt-in parallel context when eligible.
     * Unsupported layouts, small arrays and confined memory use the caller thread.
     * @param parallel reusable CPU execution context
     * @param output destination
     * @param inputs expression inputs
     * @return output
     */
    public NDArray evalInto(ParallelContext parallel, NDArray output, NDArray... inputs) {
        java.util.Objects.requireNonNull(parallel, "parallel");
        context.admit();
        var plan = new InvocationPlan(this, inputs);
        plan.checkOutput(output);
        boolean eligible = output.isCContiguous() && !plan.needsTemporary(output)
                && (output.dtype() == DType.FLOAT32 || output.dtype() == DType.FLOAT64);
        for (var input : inputs) {
            eligible &= input.isCContiguous() && input.shape().equals(output.shape())
                    && (input.dtype() == DType.FLOAT32 || input.dtype() == DType.FLOAT64);
        }
        NDArray[] arrays = java.util.Arrays.copyOf(inputs, inputs.length + 1);
        arrays[inputs.length] = output;
        NDArray[] flat = null;
        NDArray flatOutput = null;
        if (eligible) {
            flat = new NDArray[inputs.length];
            for (int i = 0; i < inputs.length; i++) flat[i] = inputs[i].reshapeView(inputs[i].size());
            flatOutput = output.reshapeView(output.size());
        }
        NDArray[] finalFlat = flat;
        NDArray finalOutput = flatOutput;
        SharedSecrets.parallelAccess().execute(parallel, arrays, output.size(), eligible, (start, end) -> {
            NDArray[] part = new NDArray[finalFlat.length];
            for (int i = 0; i < part.length; i++) part[i] = finalFlat[i].slice(Slice.range(start, end));
            evalInto(finalOutput.slice(Slice.range(start, end)), part);
        }, () -> evalInto(output, inputs));
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
        long size = output.shape().size();
        if (size == 0) { context.selected("empty", "zero elements"); return; }
        boolean small = size <= 512;
        if (context.mode() == Jit.Mode.AUTO && small && EagerExecutor.tryRun(this, inputs, output, context.eagerBytes())) {
            context.eagerRun();
            context.selected("eager", "small expression");
            return;
        }
        boolean eligible = KernelLoader.supports(this) && GeneratedExecutor.eligible(inputs, output);
        if (context.mode() == Jit.Mode.REQUIRED && !eligible)
            throw new IllegalArgumentException("generated execution does not support this layout or graph size");
        var types = new org.numjvm.dtype.DType[inputs.length];
        for (int i = 0; i < inputs.length; i++) types[i] = inputs[i].dtype();
        boolean cached = eligible && context.cached(this, types);
        boolean homogeneous = true;
        for (int i = 1; i < types.length; i++) homogeneous &= types[i] == types[0];
        boolean vectorProvider = !org.numjvm.BackendInfo.simdProvider().equals("scalar");
        // There is no intermediate traffic to save for one direct ufunc.
        if (context.mode() == Jit.Mode.AUTO && vectorProvider && EagerExecutor.isDirect(this)
                && EagerExecutor.tryRun(this, inputs, output, context.eagerBytes())) {
            context.eagerRun();
            context.selected("eager", "single ufunc");
            return;
        }
        boolean eagerCandidate = small || (size <= 65536 && homogeneous && vectorProvider && graph.nodes().size() >= 3);
        String fusedFallback = null;
        if (context.mode() == Jit.Mode.AUTO && eligible && vectorProvider && FusedProviders.provider() != null) {
            boolean fusedCached = context.fusedCached(this, types);
            if ((fusedCached && size >= 1024) || size >= 4096) {
                var fused = context.fusedKernel(this, types);
                if (fused == null) fusedFallback = context.fusedFailure();
                else if (!GeneratedExecutor.hasVectorRun(inputs, output, fused.vectorLength()))
                    fusedFallback = "runs shorter than vector length";
                else {
                    GeneratedExecutor.run(fused, inputs, output);
                    context.fusedRun();
                    context.selected("fused-vector", fusedCached ? "cached fused kernel" : "large eligible vector run");
                    return;
                }
            }
        }
        if (context.mode() == Jit.Mode.AUTO && !small && eagerCandidate
                && EagerExecutor.tryRun(this, inputs, output, context.eagerBytes())) {
            context.eagerRun();
            context.selected("eager", fusedFallback != null ? fusedFallback : !eligible ? "unsupported generated layout or graph" : "SIMD eager candidate");
            return;
        }
        // Provisional thresholds must be calibrated on the reference machine.
        boolean generate = context.mode() == Jit.Mode.REQUIRED
                || (context.mode() == Jit.Mode.AUTO && (cached || size >= 4096));
        if (generate && eligible) {
            GeneratedExecutor.run(context.kernel(this, types), inputs, output);
            context.generatedRun();
            context.selected("generated", context.mode() == Jit.Mode.REQUIRED ? "explicit mode"
                    : fusedFallback != null ? fusedFallback : cached ? "cached kernel" : "large eligible run");
        } else {
            BlockExecutor.run(this, inputs, output);
            context.blockRun();
            context.selected("block", !eligible ? "unsupported generated layout or graph"
                    : fusedFallback != null ? fusedFallback : "cold or budget-limited run");
        }
    }
}
