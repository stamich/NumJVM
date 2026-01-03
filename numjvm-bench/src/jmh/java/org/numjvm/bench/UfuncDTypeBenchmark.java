package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Element-wise ufuncs per dtype on C-contiguous operands of {@code size} elements: the
 * generated loops of each family (8-, 32- and 64-bit integers, float16 through float32,
 * float32, float64) and the hand-written complex128 loops. Integer {@code divide} and
 * {@code sqrt} include the conversion to float64; float16 computes in float32. The NumPy
 * counterparts are in {@code numpy/bench_numpy.py --suite m1}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class UfuncDTypeBenchmark {

    @Param({"1048576"})
    public long size;

    @Param({"INT8", "INT32", "INT64", "FLOAT16", "FLOAT32", "FLOAT64", "COMPLEX128"})
    public DType dtype;

    private NDArray a;
    private NDArray b;

    @Setup
    public void setup() {
        a = Layouts.make(size, Layouts.Layout.C, dtype);
        b = Layouts.make(size, Layouts.Layout.C, dtype);
    }

    @Benchmark
    public NDArray add() {
        return NJ.add(a, b);
    }

    @Benchmark
    public NDArray multiply() {
        return NJ.multiply(a, b);
    }

    @Benchmark
    public NDArray divide() {
        return NJ.divide(a, b);
    }

    @Benchmark
    public NDArray sqrt() {
        return NJ.sqrt(a);
    }

    @Benchmark
    public NDArray exp() {
        return NJ.exp(a);
    }
}
