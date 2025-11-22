package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

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
 * ContiguousRead and StridedRead: every element read once through
 * {@code getDouble(i, j)}, the public element accessor (ADR-002). With {@code -prof gc}
 * the allocation rate per operation must be zero.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class MemoryAccessBenchmark {

    @Param({"16", "128", "1024", "65536", "1048576", "16777216"})
    public long size;

    private NDArray contiguous;

    @Setup
    public void setup() {
        contiguous = Layouts.make(size, Layouts.Layout.C);
    }

    /** The non-contiguous layouts; only {@link #stridedRead} uses this state. */
    @State(Scope.Thread)
    public static class Strided {

        @Param({"16", "128", "1024", "65536", "1048576", "16777216"})
        public long size;

        @Param({"F", "TRANSPOSED", "STRIDED", "BROADCASTED"})
        public Layouts.Layout layout;

        NDArray array;

        @Setup
        public void setup() {
            array = Layouts.make(size, layout);
        }
    }

    @Benchmark
    public double contiguousRead() {
        return readAll(contiguous);
    }

    @Benchmark
    public double stridedRead(Strided s) {
        return readAll(s.array);
    }

    private static double readAll(NDArray a) {
        long rows = a.dim(0);
        long cols = a.dim(1);
        double acc = 0;
        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                acc += a.getDouble(i, j);
            }
        }
        return acc;
    }
}
