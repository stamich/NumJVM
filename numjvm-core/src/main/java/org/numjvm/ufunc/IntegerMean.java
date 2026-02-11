package org.numjvm.ufunc;

import java.lang.foreign.ValueLayout;
import java.util.function.LongToDoubleFunction;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.internal.loops.Widening;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.iteration.IterationPlan;
import org.numjvm.ndarray.NDArray;

/** Fused cast/reduce with the same virtual layout that a materialized cast would have. */
final class IntegerMean {
    private IntegerMean() { }
    static NDArray sum(NDArray x,boolean[] reduced,boolean keepdims,MemoryAllocator allocator,NDArray out) {
        if(x.isCContiguous() || x.isFContiguous()) {
            return Reducer.reduceAs("add",Widening.meanSum(x.dtype()),x,reduced,keepdims,0L,true,allocator,DType.FLOAT64,out);
        }
        long[] dims=x.shape().dimensions(),physical=x.strides().values(),virtual=new long[x.ndim()];
        int[] order=IterationPlan.memoryOrder(dims,new long[][]{physical});
        long stride=8;
        if(x.size()>0)for(int i=order.length-1;i>=0;i--) {
            virtual[order[i]]=stride;stride=Math.multiplyExact(stride,dims[order[i]]);
        }
        var access=SharedSecrets.arrayAccess();var memory=access.buffer(x).segment();long base=access.offset(x);
        LongToDoubleFunction read=offset->{
            long source=base;
            for(int d=0;d<dims.length;d++)source+=(offset/virtual[d]%dims[d])*physical[d];
            long value=Widening.read(memory,source,x.dtype());
            return x.dtype()==DType.UINT64?Unsigned.toDouble(value):(double)value;
        };
        BinaryLoop loop=(a,pa,sa,b,pb,sb,result,po,so,n)->{
            if(sa==0 && so==0) {
                result.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,po,
                        a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,pa)+pairwise(read,pb,sb,n));
            } else for(long i=0;i<n;i++)result.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,po+i*so,
                    a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,pa+i*sa)+read.applyAsDouble(pb+i*sb));
        };
        return Reducer.reduceAs("add",loop,x,reduced,keepdims,0L,true,allocator,DType.FLOAT64,out,virtual);
    }
    private static double pairwise(LongToDoubleFunction read,long offset,long stride,long n) {
        if(n<8) { double sum=0;for(long i=0;i<n;i++)sum+=read.applyAsDouble(offset+i*stride);return sum; }
        if(n>128) {long h=n/2;h-=h%8;return pairwise(read,offset,stride,h)+pairwise(read,offset+h*stride,stride,n-h);}
        double[] r=new double[8];for(int j=0;j<8;j++)r[j]=read.applyAsDouble(offset+j*stride);
        long limit=n-n%8;
        for(long i=8;i<limit;i+=8)for(int j=0;j<8;j++)r[j]+=read.applyAsDouble(offset+(i+j)*stride);
        double sum=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(long i=limit;i<n;i++)sum+=read.applyAsDouble(offset+i*stride);
        return sum;
    }
}
