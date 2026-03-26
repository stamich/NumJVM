package smoke;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Jit;

/** Consumes published core and JIT artifacts on the classpath or module path. */
public final class JitConsumer {
    private JitConsumer() { }
    public static void main(String[] args) {
        for (Jit.Mode mode : Jit.Mode.values()) {
            try (var context = Jit.context(mode)) {
                var plan = context.trace(1, i -> i.input(0).add(2).square());
                if (plan.eval(NJ.ones(DType.FLOAT64, 5)).getDouble(0) != 9) throw new AssertionError(mode);
                if (mode == Jit.Mode.REQUIRED && context.statistics().generatedRuns() != 1) throw new AssertionError("no generated execution");
            }
        }
    }
}
