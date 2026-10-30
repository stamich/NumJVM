package org.numjvm.bench;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.Ufuncs;
/** Native counters are collected outside timed JMH kernels to avoid probe overhead. */
public final class NativeAccountingProbe {
    private NativeAccountingProbe() { }
    public static void main(String[] args) {
        var a=NJ.ones(DType.INT32,1_048_576);var b=NJ.ones(1_048_576);var out=NJ.zeros(1_048_576);
        Ufuncs.ADD.apply(a,b,out);
        var before=NJ.memoryStats();
        for(int i=0;i<100;i++)Ufuncs.ADD.apply(a,b,out);
        var after=NJ.memoryStats();
        // Keep trial inputs alive until both snapshots; Cleaner activity must not
        // turn an input's release into an apparent negative scratch allocation.
        java.lang.ref.Reference.reachabilityFence(a);
        java.lang.ref.Reference.reachabilityFence(b);
        java.lang.ref.Reference.reachabilityFence(out);
        long retained=after.liveNativeBytes()-before.liveNativeBytes();
        System.out.println("{\"calls\":100,\"nativeAllocations\":"+(after.nativeAllocations()-before.nativeAllocations())+",\"retainedNativeBytes\":"+retained+"}");
        if(retained!=0 || out.getDouble(0)!=2)throw new AssertionError("scratch retained memory or wrong output");
    }
}
