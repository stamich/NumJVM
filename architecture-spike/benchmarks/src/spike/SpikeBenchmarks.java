package spike;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.TimeUnit;

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
 * S5 — JMH versions of the S1/S3 measurements.
 *
 * <p>Same operations and sizes as {@code numpy_baseline.py}. Single thread.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx2g"})
@State(Scope.Thread)
public class SpikeBenchmarks {

    @Param({"1024", "65536", "1048576"})
    public int n;

    SpikeArray a;
    SpikeArray b;
    SpikeArray bt;

    @Setup
    public void setup() {
        long side = (long) Math.sqrt(n);
        boolean nat = n * 8L > 32 * 1024;
        a = SpikeArray.arange(n, nat).reshape(side, n / side);
        b = SpikeArray.arange(n, nat).reshape(side, n / side);
        bt = SpikeArray.arange(n, nat).reshape(n / side, side).transpose();
    }

    // ------------------------------------------------------------------ allocation (S1)

    @Benchmark
    public MemorySegment allocHeap() {
        return MemorySegment.ofArray(new long[n]);
    }

    @Benchmark
    public MemorySegment allocAuto() {
        return Arena.ofAuto().allocate(n * 8L, 64);
    }

    @Benchmark
    public long allocConfined() {
        try (Arena arena = Arena.ofConfined()) {
            return arena.allocate(n * 8L, 64).byteSize();
        }
    }

    // ------------------------------------------------------------------ kernels (S3)

    @Benchmark
    public SpikeArray addContiguous() {
        return StridedAdd.add(a, b, a.segment.isNative());
    }

    @Benchmark
    public SpikeArray addGenericLoop() {
        return StridedAdd.addGeneric(a, b, a.segment.isNative());
    }

    @Benchmark
    public SpikeArray addTransposed() {
        return StridedAdd.add(a, bt, a.segment.isNative());
    }

    @Benchmark
    public double sumPairwise() {
        return SumReduction.sum(a);
    }

    @Benchmark
    public double sumNaive() {
        return SumReduction.naive(a.segment, a.offset, a.size(), 8);
    }
}
