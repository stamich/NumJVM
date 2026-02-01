package org.numjvm.linalg;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import org.numjvm.dtype.DType;

/** Checked LP64 row-major CBLAS bindings; every address is native and live for the call. */
final class NativeGemm {
    private NativeGemm() { }
    private static final MethodHandle SGEMM=bind("cblas_sgemm",ValueLayout.JAVA_FLOAT);
    private static MethodHandle bind(String name,java.lang.foreign.MemoryLayout scalar) {
        return Linker.nativeLinker().downcallHandle(OpenBlas.INSTANCE.required(name),FunctionDescriptor.ofVoid(
                ValueLayout.JAVA_INT,ValueLayout.JAVA_INT,ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT,ValueLayout.JAVA_INT,ValueLayout.JAVA_INT,
                scalar,ValueLayout.ADDRESS,ValueLayout.JAVA_INT,ValueLayout.ADDRESS,ValueLayout.JAVA_INT,
                scalar,ValueLayout.ADDRESS,ValueLayout.JAVA_INT));
    }
    static int dimension(long n) {
        if(n<0 || n>Integer.MAX_VALUE)throw new IllegalArgumentException("CBLAS LP64 dimension outside int32: "+n);
        return (int)n;
    }
    private static MemorySegment pointer(MatrixView x,long rows,long columns,DType type) {
        if(x.type()!=type || !x.memory().isNative() || x.columns()!=type.size()
                || x.rows()!=columns*type.size())throw new IllegalArgumentException("CBLAS requires packed native "+type);
        long bytes=Math.multiplyExact(Math.multiplyExact(rows,columns),type.size());
        return x.memory().asSlice(x.base(),bytes);
    }
    static void sgemm(MatrixView a,MatrixView b,MatrixView out,long rows,long contraction,long columns) {
        int m=dimension(rows),k=dimension(contraction),n=dimension(columns);
        if(m==0 || n==0)return;
        if(k==0) { for(long i=0;i<rows;i++)for(long j=0;j<columns;j++)out.set(i,j,0);return; }
        MemorySegment pa=pointer(a,m,k,DType.FLOAT32),pb=pointer(b,k,n,DType.FLOAT32),pc=pointer(out,m,n,DType.FLOAT32);
        try { SGEMM.invokeExact(101,111,111,m,n,k,1.0f,pa,Math.max(1,k),pb,Math.max(1,n),0.0f,pc,Math.max(1,n)); }
        catch(Throwable e) { throw new IllegalStateException("OpenBLAS sgemm failed; output may be partial",e); }
    }
}
