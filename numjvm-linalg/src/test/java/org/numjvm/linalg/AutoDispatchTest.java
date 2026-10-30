package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
class AutoDispatchTest {
    @Test void automaticThresholdIncludesPackingCost() {
        Assumptions.assumeTrue(OpenBlas.MODE.equals("auto") && OpenBlas.INSTANCE!=null);
        long before=Linalg.nativeCalls();
        Linalg.matmul(NJ.ones(32,32),NJ.ones(32,32),NJ.zeros(32,32));
        assertEquals(before,Linalg.nativeCalls());assertTrue(Linalg.lastBackend().startsWith("java:"));
        try(var scope=NJ.scope()) {
            var a=scope.ones(DType.FLOAT64,32,32);var out=scope.zeros(DType.FLOAT64,32,32);
            Linalg.matmul(a,a,out);assertEquals(before+1,Linalg.nativeCalls());assertEquals(32,out.getDouble(0,0));
        }
        Linalg.matmul(NJ.ones(128,128),NJ.ones(128,128),NJ.zeros(128,128));
        assertEquals(before+2,Linalg.nativeCalls());
    }
}
