package org.numjvm.bench;

import org.numjvm.BackendInfo;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.Ufuncs;

/** Reports actual vector-loop use independently from provider discovery. */
public final class ProviderProbe {
    private ProviderProbe() { }

    public static void main(String[] args) {
        var left = NJ.ones(DType.FLOAT64, 65536);
        var right = NJ.ones(DType.FLOAT64, 65536);
        var out = NJ.zeros(DType.FLOAT64, 65536);
        long before = BackendInfo.vectorRuns();
        Ufuncs.MULTIPLY.apply(left, right, out);
        long multiplyRuns = BackendInfo.vectorRuns() - before;
        before = BackendInfo.vectorRuns();
        Ufuncs.ADD.apply(left, right, out);
        long addRuns = BackendInfo.vectorRuns() - before;
        before = BackendInfo.vectorRuns();
        Ufuncs.DIVIDE.apply(left, right, out);
        long divideRuns = BackendInfo.vectorRuns() - before;
        System.out.println("NUMJVM_PROVIDER=" + BackendInfo.simdProvider());
        System.out.println("NUMJVM_PROVIDER_REASON=" + BackendInfo.simdReason());
        System.out.println("NUMJVM_VECTOR_RUNS_MULTIPLY=" + multiplyRuns);
        System.out.println("NUMJVM_VECTOR_RUNS_ADD=" + addRuns);
        System.out.println("NUMJVM_VECTOR_RUNS_DIVIDE=" + divideRuns);
    }
}
