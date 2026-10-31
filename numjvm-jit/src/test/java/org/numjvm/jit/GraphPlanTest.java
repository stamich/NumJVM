package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GraphPlanTest {
    @Test void sharedNodeIsEvaluatedOnceAndCountsBothEdges() {
        try (var context = Jit.context()) {
            Expr[] shared = new Expr[1];
            var graph = context.trace(1, inputs -> {
                shared[0] = inputs.input(0).square();
                return shared[0].add(shared[0]);
            });
            var plan = new GraphPlan(graph.root());
            assertEquals(3, plan.nodes().size());
            assertSame(graph.root(), plan.nodes().getLast());
            assertEquals(2, plan.consumers(shared[0]));
            assertEquals(0, plan.consumers(graph.root()));
        }
    }
}
