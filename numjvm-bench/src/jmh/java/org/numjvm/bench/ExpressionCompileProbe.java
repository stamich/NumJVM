package org.numjvm.bench;

import org.numjvm.NJ;
import org.numjvm.jit.Jit;

/** Reports instrumented compile time independently of tracing and first execution. */
public final class ExpressionCompileProbe {
    private ExpressionCompileProbe() { }
    public static void main(String[] args) {
        var input = NJ.ones(256);
        var output = NJ.zeros(256);
        try (var context = Jit.context(Jit.Mode.REQUIRED)) {
            long before = System.nanoTime();
            var expression = context.trace(1, i -> i.input(0).square().add(1));
            long trace = System.nanoTime() - before;
            before = System.nanoTime();
            expression.evalInto(output, input);
            long first = System.nanoTime() - before;
            long compile = context.compilationNanos();
            before = System.nanoTime();
            expression.evalInto(output, input);
            long hit = System.nanoTime() - before;
            System.out.println("traceNanos=" + trace);
            System.out.println("firstExecutionNanos=" + first);
            System.out.println("kernelCompilationNanos=" + compile);
            System.out.println("cacheHitExecutionNanos=" + hit);
            System.out.println("cacheHits=" + context.statistics().cacheHits());
            if (context.statistics().compilations() != 1 || context.statistics().cacheHits() != 1)
                throw new AssertionError("unexpected compilation or cache behavior");
        }
    }
}
