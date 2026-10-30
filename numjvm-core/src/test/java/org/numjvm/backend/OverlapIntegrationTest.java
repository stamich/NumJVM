package org.numjvm.backend;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;
class OverlapIntegrationTest {
    @Test void shiftedReversedAndBroadcastAliasesUseOriginalInputs() {
        var base=NJ.arange(0,257,1).astype(DType.FLOAT64);
        var original=base.copy();
        Ufuncs.ADD.apply(base.slice(Slice.step(-1)),NJ.ones(DType.INT8,1),base);
        for(int i=0;i<257;i++)assertEquals(original.getDouble(256-i)+1,base.getDouble(i));
        var row=base.slice(Slice.range(0,1));
        double first=row.getDouble(0);
        var old=base.copy();Ufuncs.ADD.apply(base,row,base);
        for(int i=0;i<257;i++)assertEquals(old.getDouble(i)+first,base.getDouble(i));
    }
    @Test void identicalInPlaceAndStridedOutputRemainCorrect() {
        var a=NJ.arange(0,257,1).astype(DType.FLOAT64);
        Ufuncs.ADD.apply(a,NJ.ones(DType.INT16,257),a);
        var backing=NJ.zeros(514);var out=backing.slice(Slice.step(2));
        Ufuncs.MULTIPLY.apply(a,NJ.ones(DType.FLOAT32,1).multiply(2.0),out);
        for(int i=0;i<257;i++)assertEquals(2.0*(i+1),out.getDouble(i));
    }
}
