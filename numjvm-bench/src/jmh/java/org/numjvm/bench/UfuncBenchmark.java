package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
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
 * Element-wise ufuncs on float64 operands of {@code size} elements, the first operand in each
 * layout and the second C-contiguous. Every call allocates its result, as a user call does,
 * except {@code addOut}, which writes into a preallocated array. The NumPy counterparts are
 * in {@code numpy/bench_numpy.py --suite m1} under the same names.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class UfuncBenchmark {

    @Param({"1024", "65536", "1048576"})
    public long size;

    @Param({"C", "F", "TRANSPOSED", "STRIDED", "BROADCASTED"})
    public Layouts.Layout layout;

    private NDArray a;
    private NDArray b;
    private NDArray row;
    private NDArray out;

    @Setup
    public void setup() {
        a = Layouts.make(size, layout, DType.FLOAT64);
        b = Layouts.make(size, Layouts.Layout.C, DType.FLOAT64);
        row = b.slice(Slice.index(0));
        out = NJ.zeros(a.shape().dimensions());
    }

    @Benchmark
    public NDArray add() {
        return NJ.add(a, b);
    }

    @Benchmark
    public NDArray addScalar() {
        return NJ.add(a, 1.5);
    }

    @Benchmark
    public NDArray addRow() {
        return NJ.add(a, row);
    }

    @Benchmark
    public NDArray addOut() {
        return Ufuncs.ADD.apply(a, b, out);
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
    public NDArray maximum() {
        return NJ.maximum(a, b);
    }

    @Benchmark
    public NDArray less() {
        return NJ.less(a, b);
    }

    @Benchmark
    public NDArray sqrt() {
        return NJ.sqrt(a);
    }

    @Benchmark
    public NDArray exp() {
        return NJ.exp(a);
    }

    @Benchmark
    public NDArray sin() {
        return NJ.sin(a);
    }
}
