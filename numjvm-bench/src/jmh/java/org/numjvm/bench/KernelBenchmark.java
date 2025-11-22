package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.kernels.ValidationKernels;
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
 * MiniAdd and MiniSum: the validation kernels with the first operand in each layout.
 * {@code add} includes allocating its result, as a user call does.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class KernelBenchmark {

    @Param({"16", "128", "1024", "65536", "1048576", "16777216"})
    public long size;

    @Param({"C", "F", "TRANSPOSED", "STRIDED", "BROADCASTED"})
    public Layouts.Layout layout;

    private NDArray a;
    private NDArray b;

    @Setup
    public void setup() {
        a = Layouts.make(size, layout);
        b = Layouts.make(size, Layouts.Layout.C);
    }

    @Benchmark
    public NDArray miniAdd() {
        return ValidationKernels.add(a, b);
    }

    @Benchmark
    public double miniSum() {
        return ValidationKernels.sumFloat64(a);
    }
}
