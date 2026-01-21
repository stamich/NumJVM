package org.numjvm.vector;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.BackendInfo;
import static org.junit.jupiter.api.Assertions.*;
class VectorBootstrapTest {
    @Test void installedProviderExecutesRealVectorLoopOnHeap() {
        assertEquals("jdk-vector",BackendInfo.simdProvider());
        long before=BackendInfo.vectorRuns();
        var a=NJ.arange(0,257,1).astype(org.numjvm.dtype.DType.FLOAT64);
        var out=NJ.add(a,a);
        for(int i=0;i<257;i++) assertEquals(2.0*i,out.getDouble(i));
        assertTrue(BackendInfo.vectorRuns()>before);
    }
}
