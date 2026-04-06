package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.jit.Expression;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.*;

/** Single-shot trace and first-call costs, measured separately from warm execution. */
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(3)
public class ExpressionLifecycleBenchmark {
    @State(Scope.Thread)
    public static class TraceState {
        Jit.Context context;
        @Setup(Level.Invocation) public void setup() { context = Jit.context(Jit.Mode.REQUIRED); }
        @TearDown(Level.Invocation) public void close() { context.close(); }
    }

    @State(Scope.Thread)
    public static class ColdState {
        Jit.Context context;
        Expression expression;
        NDArray input, output;
        @Setup(Level.Invocation) public void setup() {
            context = Jit.context(Jit.Mode.REQUIRED);
            expression = context.trace(1, i -> i.input(0).square().add(1));
            input = NJ.ones(256);
            output = NJ.zeros(256);
        }
        @TearDown(Level.Invocation) public void close() { context.close(); }
    }

    @State(Scope.Thread)
    public static class WarmState {
        Jit.Context context;
        Expression expression;
        NDArray input, output;
        @Setup(Level.Invocation) public void setup() {
            context = Jit.context(Jit.Mode.REQUIRED);
            expression = context.trace(1, i -> i.input(0).square().add(1));
            input = NJ.ones(256);
            output = NJ.zeros(256);
            expression.evalInto(output, input);
        }
        @TearDown(Level.Invocation) public void close() { context.close(); }
    }

    @Benchmark public Expression trace(TraceState state) {
        return state.context.trace(1, i -> i.input(0).square().add(1));
    }
    @Benchmark public NDArray firstExecution(ColdState state) {
        return state.expression.evalInto(state.output, state.input);
    }
    @Benchmark public NDArray cacheHit(WarmState state) {
        return state.expression.evalInto(state.output, state.input);
    }
}
