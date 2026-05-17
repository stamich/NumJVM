package examples;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Jit;
import org.numjvm.linalg.Linalg;

/** Demonstrates real contractions and expression strategy diagnostics. */
public final class Contractions {
    private Contractions() { }
    public static void main(String[] args) {
        try (var scope = NJ.scope(); var context = Jit.context()) {
            var matrix = scope.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
            var weights = NJ.array(new double[]{2, 3, 4});
            var dot = Linalg.dot(matrix, weights);
            var out = NJ.zeros(DType.FLOAT64, 2);
            Linalg.vecdot(matrix, weights, -1, out);
            if (dot.getDouble(0) != 20 || dot.getDouble(1) != 47 || out.getDouble(1) != 47)
                throw new AssertionError("contraction mismatch");
            if (dot.memoryOwner() != scope) throw new AssertionError("incorrect result owner");
            System.out.println("dot=" + dot + ", vecdot out=" + out);
            var expression = context.trace(1, i -> i.input(0).add(.25).square());
            var result = expression.eval(NJ.ones(DType.FLOAT64, 4099));
            if (result.getDouble(0) != 1.5625) throw new AssertionError("expression mismatch");
            System.out.println("expression strategy=" + context.lastDecision());
        }
    }
}
