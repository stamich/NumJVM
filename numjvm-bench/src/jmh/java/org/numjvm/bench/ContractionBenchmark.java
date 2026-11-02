package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.BackendInfo;
import org.numjvm.dtype.DType;
import org.numjvm.linalg.Linalg;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.*;

/** End-to-end vector contractions; setup reports the observed backend. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
@State(Scope.Thread)
public class ContractionBenchmark {
    @Param({"256", "65536", "1048576"}) public int size;
    @Param({"FLOAT32", "FLOAT64"}) public DType dtype;
    private NDArray a, b, out;

    @Setup public void setup() {
        a = NJ.ones(dtype, size);
        b = NJ.ones(dtype, size);
        out = NJ.zeros(dtype);
        Linalg.dot(a, b, out);
        System.out.println("NUMJVM_DOT_BACKEND=" + Linalg.lastBackend());
        Linalg.vecdot(a, b, out);
        System.out.println("NUMJVM_VECDOT_BACKEND=" + Linalg.lastBackend());
        System.out.println("NUMJVM_PROVIDER=" + BackendInfo.simdProvider());
        if (out.getDouble() != size) throw new AssertionError("invalid contraction result");
    }

    @Benchmark public NDArray dotAllocating() { return Linalg.dot(a, b); }
    @Benchmark public NDArray dotOut() { return Linalg.dot(a, b, out); }
    @Benchmark public NDArray vecdotAllocating() { return Linalg.vecdot(a, b); }
    @Benchmark public NDArray vecdotOut() { return Linalg.vecdot(a, b, out); }
}
