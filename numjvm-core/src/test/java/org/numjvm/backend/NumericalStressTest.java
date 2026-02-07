package org.numjvm.backend;

import java.util.Random;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class NumericalStressTest {
    @Test void conversionChunksAndUnsignedPromotionMatchMaterializedReference() {
        Random random=new Random(60128);
        for(int n:new int[]{7,8,9,127,128,129,4095,4096,4097,8193}) {
            var a=NJ.zeros(DType.UINT32,n);var b=NJ.zeros(DType.FLOAT32,n);
            for(int i=0;i<n;i++) { a.setLong(i,random.nextLong());b.setDouble(i,random.nextGaussian()); }
            var expected=NJ.add(a.astype(DType.FLOAT64),b.astype(DType.FLOAT64));
            var actual=NJ.add(a,b);
            for(int i=0;i<n;i++)assertEquals(expected.getDouble(i),actual.getDouble(i));
        }
    }
    @Test void stridedIntegerMeanPreservesCastGrouping() {
        var x=NJ.arange(0,257*4,1).reshape(257,4).slice(Slice.step(-1),Slice.step(2));
        var expected=NJ.mean(x.astype(DType.FLOAT64));
        assertEquals(expected.getDouble(),NJ.mean(x).getDouble());
    }
    @Test void integerReductionsPreserveOverflowAndNoAxisSemantics() {
        var a=NJ.array(new long[]{Long.MAX_VALUE,2,3,Long.MIN_VALUE});
        assertEquals(4,NJ.sum(a).getLong());
        var none=Ufuncs.ADD.reduce(a,new int[0],false);
        for(int i=0;i<4;i++)assertEquals(a.getLong(i),none.getLong(i));
        assertEquals(0,NJ.prod(a).getLong());
    }
}
