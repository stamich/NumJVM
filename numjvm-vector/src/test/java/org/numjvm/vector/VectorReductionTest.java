package org.numjvm.vector;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.BackendInfo;
import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.KernelRegistry;
import static org.junit.jupiter.api.Assertions.*;
class VectorReductionTest {
    @Test void orderedExtremaAndTruthReductionsActuallyUseVectors() {
        long before=BackendInfo.vectorRuns();
        var a=NJ.arange(1,258,1).astype(DType.FLOAT64);
        assertEquals(257,NJ.max(a).getDouble());assertEquals(1,NJ.min(a).getDouble());
        assertTrue(NJ.all(NJ.ones(DType.BOOL,257)).getBoolean());
        assertFalse(NJ.any(NJ.zeros(DType.BOOL,257)).getBoolean());
        assertTrue(BackendInfo.vectorRuns()>before);
    }
    @Test void integerHorizontalSumKeepsModularOverflow() {
        for(DType type:new DType[]{DType.INT32,DType.INT64}) {
            var in=MemorySegment.ofArray(new long[512]);
            for(int i=0;i<257;i++) {
                if(type==DType.INT32)in.set(ValueLayout.JAVA_INT_UNALIGNED,i*4L,Integer.MAX_VALUE);
                else in.set(ValueLayout.JAVA_LONG_UNALIGNED,i*8L,Long.MAX_VALUE);
            }
            var expected=MemorySegment.ofArray(new long[1]);var actual=MemorySegment.ofArray(new long[1]);
            var scalar=KernelRegistry.scalarBinary("add",type,type);
            scalar.run(expected,0,0,in,0,type.size(),expected,0,0,257);
            new VectorProvider().binary("add",type,scalar).run(actual,0,0,in,0,type.size(),actual,0,0,257);
            assertEquals(expected.get(ValueLayout.JAVA_LONG,0),actual.get(ValueLayout.JAVA_LONG,0));
        }
    }
}
