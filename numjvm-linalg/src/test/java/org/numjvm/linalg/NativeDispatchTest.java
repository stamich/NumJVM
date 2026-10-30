package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;
class NativeDispatchTest {
    @Test void forcedProviderActuallyExecutesForHeapAndBroadcastBatches() {
        Assumptions.assumeTrue(OpenBlas.MODE.equals("openblas"));
        for(DType t:new DType[]{DType.FLOAT32,DType.FLOAT64}) {
            var a=NJ.ones(t,2,3,4).slice(Slice.all(),Slice.step(-1),Slice.all());
            var b=NJ.ones(t,4,5);long before=Linalg.nativeCalls();
            var out=Linalg.matmul(a,b);
            assertEquals(before+2,Linalg.nativeCalls());assertEquals("openblas",Linalg.lastBackend());
            for(int i=0;i<2;i++)for(int j=0;j<3;j++)for(int k=0;k<5;k++)assertEquals(4,out.getDouble(i,j,k));
        }
    }
    @Test void nativeLimitsAreCheckedWithoutAllocatingGiganticArrays() {
        assertFalse(BlasDispatch.lp64(1,(long)Integer.MAX_VALUE+1,1));
        assertTrue(BlasDispatch.lp64(1,Integer.MAX_VALUE,1));
    }
    @Test void emptyContractionNeedsNoDowncall() {
        var out=Linalg.matmul(NJ.zeros(3,0),NJ.zeros(0,2));
        assertEquals(0,out.getDouble(0,0));assertTrue(Linalg.lastBackend().startsWith("java:"));
    }
}
