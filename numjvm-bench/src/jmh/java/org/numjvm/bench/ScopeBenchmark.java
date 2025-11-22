package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * ScopeAllocation: one array allocated in an open scope. To keep memory bounded the
 * scope is closed and reopened whenever it holds 256 MiB, which adds that close cost,
 * amortised, to large sizes. ScopeClose: open a scope, allocate one array, close it.
 * Both run for confined and shared scopes.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=16g"})
@State(Scope.Thread)
public class ScopeBenchmark {

    @Param({"16", "128", "1024", "65536", "1048576", "16777216"})
    public long size;

    @Param({"CONFINED", "SHARED"})
    public NJScope.Mode mode;

    private static final long RECYCLE_BYTES = 256L << 20;

    private NJScope scope;
    private long scopeBytes;

    @Setup(Level.Iteration)
    public void open() {
        scope = newScope();
        scopeBytes = 0;
    }

    private NJScope newScope() {
        return mode == NJScope.Mode.CONFINED ? NJ.scope() : NJ.scopeShared();
    }

    @TearDown(Level.Iteration)
    public void close() {
        scope.close();
    }

    @Benchmark
    public NDArray scopeAllocation() {
        scopeBytes += size * 8;
        if (scopeBytes > RECYCLE_BYTES) {
            scope.close();
            scope = newScope();
            scopeBytes = size * 8;
        }
        return scope.zeros(DType.FLOAT64, size);
    }

    @Benchmark
    public long scopeClose() {
        try (NJScope s = newScope()) {
            return s.zeros(DType.FLOAT64, size).size();
        }
    }
}
