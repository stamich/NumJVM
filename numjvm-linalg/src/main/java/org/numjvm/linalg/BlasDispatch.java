package org.numjvm.linalg;

import java.util.concurrent.atomic.LongAdder;
import org.numjvm.dtype.DType;

/** Chooses before execution; native failures are never caught and retried. */
final class BlasDispatch {
    private static final LongAdder CALLS=new LongAdder();
    private static final ThreadLocal<String> LAST=ThreadLocal.withInitial(()->"not executed on this thread");
    private BlasDispatch() { }
    static void initialize() { OpenBlas.availability(); }
    static long calls() { return CALLS.sum(); }
    static String last() { return LAST.get(); }
    static void emptyBatch() { LAST.set("java: empty batch/output; no arithmetic executed"); }
    static boolean lp64(long m,long k,long n) {
        return m>=0 && k>=0 && n>=0 && m<=Integer.MAX_VALUE && k<=Integer.MAX_VALUE && n<=Integer.MAX_VALUE;
    }
    static void run(MatrixView a,MatrixView b,MatrixView out,long m,long k,long n) {
        boolean nativeCall=OpenBlas.INSTANCE!=null && lp64(m,k,n) && m>0 && k>0 && n>0
                && (OpenBlas.MODE.equals("openblas") || autoEligible(a,b,out,m,k,n));
        if(!nativeCall) {
            LAST.set("java: "+(OpenBlas.INSTANCE==null?OpenBlas.availability():!lp64(m,k,n)?"outside LP64 limits":"small or empty contraction"));
            JavaMatmul.blocked(a,b,out,m,k,n);return;
        }
        try(var pa=Operand.prepare(a,m,k,out.type());var pb=Operand.prepare(b,k,n,out.type());
            var pc=Operand.output(out,m,n)) {
            if(out.type()==DType.FLOAT32) NativeGemm.sgemm(pa.view,pb.view,pc.view,m,k,n);
            else NativeGemm.dgemm(pa.view,pb.view,pc.view,m,k,n);
            CALLS.increment(); LAST.set("openblas");
            if(pc.storage!=null)for(long i=0;i<m;i++)for(long j=0;j<n;j++)out.set(i,j,pc.view.get(i,j));
        }
    }
    private static boolean autoEligible(MatrixView a,MatrixView b,MatrixView out,long m,long k,long n) {
        // Three-fork samples show packing erases the benefit at 32^3. Keep that
        // case in Java; already-native matrices avoid packing and cross over earlier.
        boolean direct=Operand.compatible(a,k,out.type()) && Operand.compatible(b,n,out.type())
                && Operand.compatible(out,n,out.type());
        int minimum=direct?32:128;
        return m>=minimum && k>=minimum && n>=minimum;
    }
    private static final class Operand implements AutoCloseable {
        final MatrixView view;
        final PackedMatrix storage;
        private Operand(MatrixView view,PackedMatrix storage) { this.view=view;this.storage=storage; }
        private static boolean compatible(MatrixView v,long columns,DType type) {
            return v.type()==type && v.memory().isNative() && v.columns()==type.size() && v.rows()==columns*type.size();
        }
        static Operand prepare(MatrixView v,long rows,long columns,DType type) {
            if(compatible(v,columns,type))return new Operand(v,null);
            var pack=PackedMatrix.pack(v,rows,columns,type);return new Operand(pack.view(),pack);
        }
        static Operand output(MatrixView v,long rows,long columns) {
            if(compatible(v,columns,v.type()))return new Operand(v,null);
            var pack=PackedMatrix.allocate(rows,columns,v.type());return new Operand(pack.view(),pack);
        }
        @Override public void close() { if(storage!=null)storage.close(); }
    }
}
