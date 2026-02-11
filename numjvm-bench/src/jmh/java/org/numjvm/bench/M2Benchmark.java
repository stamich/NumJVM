package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Reductions;
import org.numjvm.ufunc.Ufuncs;
import org.openjdk.jmh.annotations.*;

/** Allocation/conversion checkpoints; allocating and preallocated calls stay separate. */
@BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations=3,time=1) @Measurement(iterations=5,time=1) @Fork(3) @State(Scope.Thread)
public class M2Benchmark {
    @Param({"1024","65536","1048576"}) public int size;
    private NDArray integers,floats,out,sumOut,meanOut;
    @Setup public void setup() {
        integers=NJ.arange(0,size,1).astype(DType.INT32);
        floats=NJ.ones(size);out=NJ.zeros(size);sumOut=NJ.zeros(DType.INT64);meanOut=NJ.zeros();
    }
    @Benchmark public NDArray mixedAdd() { return NJ.add(integers,floats); }
    @Benchmark public NDArray mixedAddOut() { return Ufuncs.ADD.apply(integers,floats,out); }
    @Benchmark public NDArray wideningSum() { return NJ.sum(integers); }
    @Benchmark public NDArray wideningSumOut() { return Reductions.sum(integers,null,false,sumOut); }
    @Benchmark public NDArray integerMean() { return NJ.mean(integers); }
    @Benchmark public NDArray integerMeanOut() { return Reductions.mean(integers,null,false,meanOut); }
}
