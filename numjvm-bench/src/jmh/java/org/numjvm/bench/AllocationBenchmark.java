package org.numjvm.bench;

import java.util.concurrent.TimeUnit;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.HeapAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.internal.memory.NativeAllocator;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Shape;
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
 * ArrayAllocation: {@code NJ.zeros} through the default hybrid allocator.
 * HeapVsNative: the same allocation forced onto each path.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
@State(Scope.Thread)
public class AllocationBenchmark {

    @Param({"16", "128", "1024", "65536", "1048576", "16777216"})
    public long size;

    @Param({"HEAP", "NATIVE"})
    public String memory;

    private MemoryAllocator allocator;
    private Shape shape;

    @Setup
    public void setup() {
        allocator = memory.equals("HEAP") ? HeapAllocator.INSTANCE : NativeAllocator.automatic();
        shape = Shape.of(size);
    }

    @Benchmark
    public NDArray arrayAllocation() {
        return NJ.zeros(DType.FLOAT64, size);
    }

    @Benchmark
    public NDArray heapVsNative() {
        return SharedSecrets.arrayAccess().allocate(allocator, DType.FLOAT64, shape, Order.C);
    }
}
