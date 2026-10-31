package smoke;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Jit;
import org.numjvm.linalg.Linalg;

/** Consumer of published artifacts, independent from the project build. */
public final class Consumer {
    private Consumer() { }
    public static void main(String[] args) {
        boolean expectFused = Boolean.getBoolean("numjvm.expect.fused");
        try (var context = Jit.context(Jit.Mode.AUTO)) {
            var graph = context.trace(1, i -> i.input(0).add(.25).square());
            var output = graph.eval(NJ.ones(DType.FLOAT64, 4099));
            if (output.getDouble(4098) != 1.5625) throw new AssertionError("expression result");
            boolean fused = context.lastDecision().strategy().equals("fused-vector");
            if (fused != expectFused) throw new AssertionError("unexpected fused selection: " + context.lastDecision());
        }
        try (var scope = NJ.scope()) {
            var a = scope.ones(DType.FLOAT64, 2048);
            var b = scope.ones(DType.FLOAT64, 2048);
            if (Linalg.dot(a, b).getDouble() != 2048 || Linalg.vecdot(a, b).getDouble() != 2048)
                throw new AssertionError("contraction result");
            if (Boolean.getBoolean("numjvm.expect.native") && !Linalg.lastBackend().equals("openblas-dot"))
                throw new AssertionError("OpenBLAS DOT was not executed: " + Linalg.lastBackend());
        }
    }
}
