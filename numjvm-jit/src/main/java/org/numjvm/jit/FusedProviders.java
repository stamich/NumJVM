package org.numjvm.jit;

import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.FusedGraph;
import org.numjvm.internal.backend.FusedKernel;
import org.numjvm.internal.backend.FusedProvider;

/** Optional service discovery; missing or unresolved vector modules remain usable. */
final class FusedProviders {
    private static final FusedProvider PROVIDER = load();
    private FusedProviders() { }

    private static FusedProvider load() {
        try { return ServiceLoader.load(FusedProvider.class).findFirst().orElse(null); }
        catch (ServiceConfigurationError | LinkageError unavailable) { return null; }
    }
    static FusedProvider provider() { return PROVIDER; }
    static String name() { return PROVIDER == null ? "unavailable" : PROVIDER.name(); }
    static FusedKernel compile(Expression expression, DType[] inputs) {
        if (PROVIDER == null) return null;
        var types = new TypeResolver(inputs);
        var nodes = new ArrayList<FusedGraph.Node>();
        var ids = new IdentityHashMap<Expr, Integer>();
        for (Expr node : expression.graph().nodes()) {
            int left = node.left == null ? -1 : ids.get(node.left);
            int right = node.right == null ? -1 : ids.get(node.right);
            ids.put(node, nodes.size());
            nodes.add(new FusedGraph.Node(FusedGraph.Op.valueOf(node.op.name()), types.resolve(node),
                    node.slot, node.bits, left, right));
        }
        return PROVIDER.compile(new FusedGraph(nodes, List.of(inputs.clone())));
    }
}
