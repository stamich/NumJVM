package org.numjvm.linalg;

import org.numjvm.dtype.DType;

/** Portable reference contraction; optimized kernels must satisfy the same numerical contract. */
final class JavaMatmul {
    private JavaMatmul() { }
    static void reference(MatrixView a,MatrixView b,MatrixView out,long m,long k,long n) {
        for(long i=0;i<m;i++) for(long j=0;j<n;j++) {
            if(out.type()==DType.FLOAT32) {
                float value=0;
                for(long h=0;h<k;h++) value+=(float)a.get(i,h)*(float)b.get(h,j);
                out.set(i,j,value);
            } else {
                double value=0;
                for(long h=0;h<k;h++) value+=a.get(i,h)*b.get(h,j);
                out.set(i,j,value);
            }
        }
    }
}
