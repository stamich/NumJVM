package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;
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
 * ViewCreation, Reshape, Transpose, BroadcastCreation: metadata-only operations, whose
 * cost must not depend on the element count.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class ViewBenchmark {

    @Param({"16", "128", "1024", "65536", "1048576", "16777216"})
    public long size;

    private NDArray matrix;
    private NDArray row;
    private long rows;
    private long cols;

    @Setup
    public void setup() {
        matrix = Layouts.make(size, Layouts.Layout.C);
        rows = Layouts.rows(size);
        cols = Layouts.cols(size);
        row = Layouts.make(cols, Layouts.Layout.C).reshapeView(cols);
    }

    @Benchmark
    public NDArray viewCreation() {
        return matrix.slice(Slice.range(0, rows, 2), Slice.step(-1));
    }

    @Benchmark
    public NDArray reshape() {
        return matrix.reshapeView(cols, rows);
    }

    @Benchmark
    public NDArray transpose() {
        return matrix.transpose();
    }

    @Benchmark
    public NDArray broadcastCreation() {
        return row.broadcastTo(rows, cols);
    }
}
