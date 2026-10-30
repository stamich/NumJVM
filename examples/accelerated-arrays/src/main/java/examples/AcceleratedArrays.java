package examples;

import org.numjvm.NJ;
import org.numjvm.BackendInfo;

/** The same application runs with an optional provider or on the scalar fallback. */
public final class AcceleratedArrays {
    private AcceleratedArrays() { }
    public static void main(String[] args) {
        var a=NJ.ones(4097);
        long before=BackendInfo.vectorRuns();
        var result=NJ.add(a,a);
        if(result.getDouble(4096)!=2)throw new AssertionError("wrong vector/tail result");
        if(BackendInfo.simdProvider().equals("jdk-vector") && BackendInfo.vectorRuns()<=before) {
            throw new AssertionError("installed provider did not execute");
        }
        System.out.println("provider="+BackendInfo.simdProvider()+", reason="+BackendInfo.simdReason()
                +", vector runs="+(BackendInfo.vectorRuns()-before));
    }
}
