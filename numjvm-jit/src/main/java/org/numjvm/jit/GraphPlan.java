package org.numjvm.jit;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** A topological plan and edge counts for releasing intermediates after use. */
final class GraphPlan {
    private record Visit(Expr node, boolean after) { }
    private final List<Expr> nodes;
    private final Map<Expr, Integer> consumers;

    GraphPlan(Expr root) {
        ArrayList<Expr> ordered = new ArrayList<>();
        IdentityHashMap<Expr, Integer> uses = new IdentityHashMap<>();
        Set<Expr> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        ArrayDeque<Visit> pending = new ArrayDeque<>();
        pending.push(new Visit(root, false));
        while (!pending.isEmpty()) {
            Visit visit = pending.pop();
            Expr node = visit.node();
            if (visit.after()) {
                ordered.add(node);
                continue;
            }
            if (!visited.add(node)) continue;
            pending.push(new Visit(node, true));
            if (node.right != null) {
                uses.merge(node.right, 1, Integer::sum);
                pending.push(new Visit(node.right, false));
            }
            if (node.left != null) {
                uses.merge(node.left, 1, Integer::sum);
                pending.push(new Visit(node.left, false));
            }
        }
        nodes = List.copyOf(ordered);
        consumers = Collections.unmodifiableMap(uses);
    }

    List<Expr> nodes() { return nodes; }
    int consumers(Expr node) { return consumers.getOrDefault(node, 0); }
}
