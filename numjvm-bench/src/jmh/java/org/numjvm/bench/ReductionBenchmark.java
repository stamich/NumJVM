package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.NJ;
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
 * Sums over every layout (T4.5): all elements, and along each of the two axes of a
 * {@code rows x cols} float64 array. The M1 exit criterion asks for every layout within 2x of
 * C-contiguous.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class ReductionBenchmark {

    @Param({"1024", "65536", "1048576"})
    public long size;

    @Param({"C", "F", "TRANSPOSED", "STRIDED", "BROADCASTED"})
    public Layouts.Layout layout;

    private NDArray a;

    @Setup
    public void setup() {
        a = Layouts.make(size, layout);
    }

    @Benchmark
    public NDArray sumAll() {
        return NJ.sum(a);
    }

    @Benchmark
    public NDArray sumAxis0() {
        return NJ.sum(a, 0);
    }

    @Benchmark
    public NDArray sumAxis1() {
        return NJ.sum(a, 1);
    }

    @Benchmark
    public NDArray maxAll() {
        return NJ.max(a);
    }
}
