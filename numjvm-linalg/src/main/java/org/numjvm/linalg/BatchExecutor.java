package org.numjvm.linalg;

import org.numjvm.iteration.IndexIterator;
import org.numjvm.ndarray.NDArray;

/** Synchronous batch broadcasting; NumJVM never starts a matrix worker pool. */
final class BatchExecutor {
    private BatchExecutor() { }
    @FunctionalInterface interface Kernel {
        void run(MatrixView a,MatrixView b,MatrixView out,long m,long k,long n);
    }
    static void execute(NDArray a,NDArray b,NDArray out,MatmulPlan p,Kernel kernel) {
        MatrixView av=MatrixView.input(a,true),bv=MatrixView.input(b,false),ov=MatrixView.output(out,p);
        var iterator=new IndexIterator(p.batch());
        while(iterator.hasNext()) {
            long[] index=iterator.next();
            long oa=offset(a,index,Math.max(0,a.ndim()-2));
            long ob=offset(b,index,Math.max(0,b.ndim()-2));
            long oo=0;
            for(int i=0;i<index.length;i++)oo+=index[i]*out.strides().stride(i);
            kernel.run(av.at(oa),bv.at(ob),ov.at(oo),p.m(),p.k(),p.n());
        }
    }
    private static long offset(NDArray a,long[] index,int batchRank) {
        long offset=0;
        for(int i=0;i<batchRank;i++)if(a.dim(i)!=1)offset+=index[index.length-batchRank+i]*a.strides().stride(i);
        return offset;
    }
}
