package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.BackendInfo;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Expression;
import org.numjvm.jit.Jit;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ufunc.Ufuncs;
import org.openjdk.jmh.annotations.*;

/** Warm expression comparisons; unsupported generated layouts fail explicitly. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class ExpressionMatrixBenchmark {
    @Param({"256", "65536", "1048576"}) public int size;
    @Param({"C"}) public Layouts.Layout layout;
    @Param({"FLOAT32", "FLOAT64", "MIXED"}) public String types;
    @Param({"ADD", "CHAIN"}) public String chain;

    private NDArray a, b, c, d, out, temp1, temp2;
    private NDArray[] args;
    private Jit.Context auto, block, generated, scheduled;
    private Expression autoPlan, blockPlan, generatedPlan, scheduledPlan;

    @Setup public void setup() {
        DType leftType = types.equals("FLOAT32") ? DType.FLOAT32 : DType.FLOAT64;
        DType rightType = types.equals("FLOAT64") ? DType.FLOAT64 : DType.FLOAT32;
        a = Layouts.make(size, layout, leftType);
        b = Layouts.make(size, layout, rightType);
        c = Layouts.make(size, layout, rightType);
        d = Layouts.make(size, layout, rightType);
        args = chain.equals("ADD") ? new NDArray[]{a, b} : new NDArray[]{a, b, c, d};
        DType resultType = leftType == DType.FLOAT64 || rightType == DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32;
        out = NJ.zeros(resultType, Layouts.rows(size), Layouts.cols(size));
        if (layout == Layouts.Layout.F) out = out.copy(Order.F);
        temp1 = NJ.zeros(resultType, Layouts.rows(size), Layouts.cols(size));
        temp2 = NJ.zeros(resultType, Layouts.rows(size), Layouts.cols(size));
        auto = Jit.context(Jit.Mode.AUTO); block = Jit.context(Jit.Mode.BLOCK); generated = Jit.context(Jit.Mode.REQUIRED);
        scheduled = Jit.context(Jit.Mode.EAGER, 64, 65536, 64L * 1024 * 1024);
        autoPlan = trace(auto); blockPlan = trace(block); generatedPlan = trace(generated); scheduledPlan = trace(scheduled);
        NDArray expected = eagerAllocating();
        check(expected, eagerOut());
        check(expected, blockOut());
        long before = BackendInfo.vectorRuns();
        check(expected, autoOut());
        System.out.println("NUMJVM_AUTO_DECISION=" + auto.lastDecision());
        System.out.println("NUMJVM_PROVIDER=" + BackendInfo.simdProvider());
        System.out.println("NUMJVM_AUTO_VECTOR_RUNS=" + (BackendInfo.vectorRuns() - before));
        check(expected, scheduledOut());
        if (layout == Layouts.Layout.C || layout == Layouts.Layout.F) check(expected, generatedOut());
        checkSpecialValues(leftType, rightType);
    }

    private Expression trace(Jit.Context context) {
        if (chain.equals("ADD")) return context.trace(2, i -> i.input(0).add(i.input(1)));
        return context.trace(4, i -> i.input(0).multiply(i.input(1)).add(i.input(2)).divide(i.input(3)));
    }

    private void check(NDArray expected, NDArray actual) {
        for (long i = 0; i < Layouts.rows(size); i++) for (long j = 0; j < Layouts.cols(size); j++) {
            double x = expected.getDouble(i, j), y = actual.getDouble(i, j);
            if (Double.doubleToLongBits(x) != Double.doubleToLongBits(y))
                throw new AssertionError("expression mismatch at " + i + "," + j + ": " + x + " != " + y);
        }
    }

    private static void checkSpecialValues(DType leftType, DType rightType) {
        var left = NJ.array(new double[]{-0.0, 0.0, Double.NaN, Double.POSITIVE_INFINITY, 2.0}).astype(leftType);
        var right = NJ.array(new double[]{1.0, -1.0, 2.0, 0.0, 0.0}).astype(rightType);
        var expected = NJ.divide(NJ.add(NJ.multiply(left, right), right), right);
        for (var mode : Jit.Mode.values()) {
            try (var context = Jit.context(mode)) {
                var plan = context.trace(2, i -> i.input(0).multiply(i.input(1)).add(i.input(1)).divide(i.input(1)));
                var actual = plan.eval(left, right);
                for (int n = 0; n < 5; n++) {
                    if (Double.doubleToLongBits(expected.getDouble(n)) != Double.doubleToLongBits(actual.getDouble(n)))
                        throw new AssertionError("special-value expression mismatch for " + mode + " at " + n);
                }
            }
        }
    }

    @Benchmark public NDArray eagerAllocating() {
        return chain.equals("ADD") ? NJ.add(a, b) : NJ.divide(NJ.add(NJ.multiply(a, b), c), d);
    }
    @Benchmark public NDArray eagerOut() {
        if (chain.equals("ADD")) return Ufuncs.ADD.apply(a, b, out);
        Ufuncs.MULTIPLY.apply(a, b, temp1);
        Ufuncs.ADD.apply(temp1, c, temp2);
        return Ufuncs.DIVIDE.apply(temp2, d, out);
    }
    @Benchmark public NDArray autoAllocating() { return autoPlan.eval(args); }
    @Benchmark public NDArray autoOut() { return autoPlan.evalInto(out, args); }
    @Benchmark public NDArray blockAllocating() { return blockPlan.eval(args); }
    @Benchmark public NDArray blockOut() { return blockPlan.evalInto(out, args); }
    @Benchmark public NDArray scheduledOut() { return scheduledPlan.evalInto(out, args); }
    @Benchmark public NDArray generatedAllocating() { return generatedPlan.eval(args); }
    @Benchmark public NDArray generatedOut() { return generatedPlan.evalInto(out, args); }
    @TearDown public void close() { auto.close(); block.close(); generated.close(); scheduled.close(); }
}
