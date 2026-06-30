package examples;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;

/** Explicitly traced expressions can be reused with different input arrays. */
public final class FusedExpressions {
    private FusedExpressions() { }

    public static void main(String[] args) {
        for (var mode : Jit.Mode.values()) {
          try (var context = Jit.context(mode)) {
            var expression = context.trace(2, i -> i.input(0).multiply(i.input(1)).add(2).sqrt());
            NDArray kept;
            try (var scope = NJ.scope()) {
                var a = scope.ones(DType.FLOAT64, 4);
                var b = NJ.ones(DType.FLOAT64, 4);
                var out = scope.zeros(DType.FLOAT64, 4);
                expression.evalInto(out, a, b);
                kept = scope.keep(out);
            }
            if (Math.abs(kept.getDouble(0) - Math.sqrt(3)) > 1e-12) throw new AssertionError("expression result");
            System.out.println(kept);
            System.out.println(mode + " " + context.statistics());
          }
        }
        // Automatic memory is accessible to the context's workers.
        try (var pool = NJ.parallel(4); var context = Jit.context(Jit.Mode.AUTO)) {
            var expression = context.trace(2, i -> i.input(0).add(i.input(1)).square());
            var a = NJ.ones(DType.FLOAT32, 65536);
            var b = NJ.ones(DType.FLOAT32, 65536);
            var out = NJ.zeros(DType.FLOAT32, 65536);
            expression.evalInto(pool, out, a, b);
            if (out.getDouble(0) != 4) throw new AssertionError("parallel expression result");
            System.out.println("parallel sum=" + pool.sum(out).getDouble());
            System.out.println("execution=" + pool.lastDecision());
        }
    }
}
