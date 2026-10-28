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
 * Whole-array reductions per dtype on a C-contiguous array of {@code size} elements: sum
 * (pairwise for floats, int64 accumulation for small integers), mean, max and any. The NumPy
 * counterparts are in {@code numpy/bench_numpy.py --suite m1}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class ReductionDTypeBenchmark {

    @Param({"1048576"})
    public long size;

    @Param({"INT8", "INT32", "INT64", "FLOAT16", "FLOAT32", "FLOAT64", "COMPLEX128"})
    public DType dtype;

    private NDArray a;

    @Setup
    public void setup() {
        a = Layouts.make(size, Layouts.Layout.C, dtype);
    }

    @Benchmark
    public NDArray sum() {
        return NJ.sum(a);
    }

    @Benchmark
    public NDArray mean() {
        return NJ.mean(a);
    }

    /**
     * The maximum; complex128 has no ordering, so its row measures the max of the real parts
     * through a strided float64 view.
     *
     * @return the maximum
     */
    @Benchmark
    public NDArray max() {
        return NJ.max(dtype.isComplex() ? a.real() : a);
    }

    @Benchmark
    public NDArray any() {
        return NJ.any(a);
    }
}
