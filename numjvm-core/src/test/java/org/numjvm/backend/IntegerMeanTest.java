package org.numjvm.backend;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
class IntegerMeanTest {
    @Test void fusedConversionKeepsReferenceBitsAtPairwiseBoundaries() {
        for (DType t : new DType[]{DType.INT32,DType.INT64,DType.UINT64}) {
            for (int n : new int[]{0,1,7,8,9,127,128,129,257}) {
                var a = NJ.zeros(t,n);
                for (int i=0;i<n;i++) { a.setLong(i,(i%3==0 ? Long.MAX_VALUE : i%3==1 ? Long.MIN_VALUE : 3)); }
                double expected = NJ.mean(a.astype(DType.FLOAT64)).getDouble();
                double actual = NJ.mean(a).getDouble();
                assertEquals(Double.doubleToLongBits(expected),Double.doubleToLongBits(actual),t+" n="+n);
            }
        }
    }
    @Test void columnAndTransposedMeansKeepTheReferenceOrder() {
        var a=NJ.arange(0,1024,1).reshape(16,64).transpose();
        for(int axis:new int[]{0,1}) {
            var expected=NJ.mean(a.astype(DType.FLOAT64),axis); var actual=NJ.mean(a,axis);
            for(long i=0;i<actual.size();i++) assertEquals(expected.getDouble(i),actual.getDouble(i));
        }
    }
    @Test void virtualCastLayoutHandlesCancellationAcrossStridedAxes() {
        var a=NJ.zeros(DType.INT64,33,257);
        for(int i=0;i<33;i++)for(int j=0;j<257;j++)a.setLong(i,j,j%3==0?Long.MAX_VALUE:j%3==1?Long.MIN_VALUE:3);
        a=a.transpose().slice(org.numjvm.ndarray.Slice.step(-2),org.numjvm.ndarray.Slice.all());
        for(int axis:new int[]{0,1}) {
            var expected=NJ.mean(a.astype(DType.FLOAT64),axis);var actual=NJ.mean(a,axis);
            for(long i=0;i<actual.size();i++)assertEquals(Double.doubleToLongBits(expected.getDouble(i)),Double.doubleToLongBits(actual.getDouble(i)));
        }
    }
}
