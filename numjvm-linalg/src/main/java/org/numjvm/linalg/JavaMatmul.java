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
    static void blocked(MatrixView a, MatrixView b, MatrixView out, long m, long k, long n) {
        if (out.type() == DType.FLOAT32) { blockedFloat(a,b,out,m,k,n); }
        else { blockedDouble(a,b,out,m,k,n); }
    }
    private static void blockedFloat(MatrixView a,MatrixView b,MatrixView out,long m,long k,long n) {
        // Keep the contraction index monotonic; blocking changes locality, not arithmetic order.
        final int block = 32;
        float[] tile = new float[block * block];
        for(long ib=0;ib<m;ib+=block) for(long jb=0;jb<n;jb+=block) {
            int height=(int)Math.min(block,m-ib),width=(int)Math.min(block,n-jb);
            java.util.Arrays.fill(tile,(float)0);
            for(long kb=0;kb<k;kb+=block) {
                long end=Math.min(k,kb+block);
                for(int i=0;i<height;i++) for(long h=kb;h<end;h++) {
                    float x=(float)a.get(ib+i,h);
                    for(int j=0;j<width;j++) tile[i*block+j]+=x*(float)b.get(h,jb+j);
                }
            }
            for(int i=0;i<height;i++) for(int j=0;j<width;j++) out.set(ib+i,jb+j,tile[i*block+j]);
        }
    }
    private static void blockedDouble(MatrixView a,MatrixView b,MatrixView out,long m,long k,long n) {
        // Keep the contraction index monotonic; blocking changes locality, not arithmetic order.
        final int block = 32;
        double[] tile = new double[block * block];
        for(long ib=0;ib<m;ib+=block) for(long jb=0;jb<n;jb+=block) {
            int height=(int)Math.min(block,m-ib),width=(int)Math.min(block,n-jb);
            java.util.Arrays.fill(tile,(double)0);
            for(long kb=0;kb<k;kb+=block) {
                long end=Math.min(k,kb+block);
                for(int i=0;i<height;i++) for(long h=kb;h<end;h++) {
                    double x=a.get(ib+i,h);
                    for(int j=0;j<width;j++) tile[i*block+j]+=x*b.get(h,jb+j);
                }
            }
            for(int i=0;i<height;i++) for(int j=0;j<width;j++) out.set(ib+i,jb+j,tile[i*block+j]);
        }
    }
}
