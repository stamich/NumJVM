package org.numjvm.bench;

import java.util.concurrent.TimeUnit;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.linalg.Linalg;
import org.numjvm.ndarray.NDArray;
import org.openjdk.jmh.annotations.*;

/** End-to-end matrix costs, including packing; external BLAS threads must be fixed by the runner. */
@BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations=3,time=1) @Measurement(iterations=5,time=1) @Fork(3) @State(Scope.Thread)
public class MatmulBenchmark {
    @Param({"32x32x32","256x64x128","1024x1024x1024","1x256x16"}) public String shape;
    @Param({"HEAP","AUTOMATIC_NATIVE","TRANSPOSED"}) public String layout;
    private NDArray a,b,out;
    @Setup public void setup() {
        String[] dims=shape.split("x");int m=Integer.parseInt(dims[0]),k=Integer.parseInt(dims[1]),n=Integer.parseInt(dims[2]);
        if(layout.equals("AUTOMATIC_NATIVE")) {
            a=nativeArray(m,k);b=nativeArray(k,n);out=nativeArray(m,n);
        } else {
            a=layout.equals("TRANSPOSED")?NJ.ones(k,m).transpose():NJ.ones(m,k);
            b=NJ.ones(k,n);out=NJ.zeros(m,n);
        }
    }
    private static NDArray nativeArray(int rows,int columns) {
        var array=org.numjvm.internal.SharedSecrets.arrayAccess().allocate(
                org.numjvm.internal.memory.NativeAllocator.automatic(),DType.FLOAT64,
                org.numjvm.ndarray.Shape.of(rows,columns),org.numjvm.ndarray.Order.C);
        org.numjvm.ufunc.Ufuncs.ADD.apply(array,1.0,array);
        return array;
    }
    @Benchmark public NDArray allocating() { return Linalg.matmul(a,b); }
    @Benchmark public NDArray preallocated() { return Linalg.matmul(a,b,out); }
}
