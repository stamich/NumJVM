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
    }
}
