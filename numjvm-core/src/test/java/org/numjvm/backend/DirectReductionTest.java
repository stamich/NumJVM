package org.numjvm.backend;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import org.numjvm.ufunc.Reductions;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;
class DirectReductionTest {
    @Test void directAxisOutputAndOverlapFallback() {
        var a=NJ.arange(0,24,1).reshape(4,6);
        var out=NJ.zeros(DType.INT64,6);
        assertSame(out,Reductions.sum(a,new int[]{0},false,out));
        var expected=NJ.sum(a,0);
        for(int i=0;i<6;i++) assertEquals(expected.getLong(i),out.getLong(i));
        var alias=a.slice(Slice.index(0));
        Reductions.sum(a,new int[]{0},false,alias);
        for(int i=0;i<6;i++) assertEquals(expected.getLong(i),alias.getLong(i));
    }
    @Test void meanWritesIntoUserOutputAndStridedFallback() {
        var a=NJ.arange(0,24,1).reshape(4,6);
        var expected=NJ.mean(a,0); var out=NJ.zeros(6);
        assertSame(out,Reductions.mean(a,new int[]{0},false,out));
        for(int i=0;i<6;i++) assertEquals(expected.getDouble(i),out.getDouble(i));
        var backing=NJ.zeros(12);var strided=backing.slice(Slice.step(2));
        Reductions.mean(a,new int[]{0},false,strided);
        for(int i=0;i<6;i++) assertEquals(expected.getDouble(i),strided.getDouble(i));
    }
    @Test void rejectsOutputBeforeDoingWork() {
        var a=NJ.array(new double[]{1,2,3});var out=NJ.array(new double[]{42,43});
        assertThrows(org.numjvm.exception.ShapeException.class,()->Ufuncs.ADD.reduce(a,null,false,out));
        assertEquals(42,out.getDouble(0));
    }
    @Test void lifecycleValidationDoesNotOverwriteOutputIdentity() {
        var scope=NJ.scope();var a=scope.ones(DType.FLOAT64,3);scope.close();
        var out=NJ.array(new double[]{42}).reshape();
        assertThrows(org.numjvm.exception.MemoryException.class,()->Ufuncs.ADD.reduce(a,null,false,out));
        assertEquals(42,out.getDouble());
    }
    @Test void floatingMeanOutputKeepsTheAllocatingReferenceBits() {
        for(DType type:new DType[]{DType.FLOAT32,DType.FLOAT64}) {
            var a=NJ.arange(0,257*3,1).astype(type).reshape(257,3).transpose();
            for(int axis:new int[]{0,1}) {
                var expected=NJ.mean(a,axis);var out=NJ.zeros(type,expected.shape().dimensions());
                Reductions.mean(a,new int[]{axis},false,out);
                for(long i=0;i<out.size();i++)assertEquals(Double.doubleToLongBits(expected.getDouble(i)),Double.doubleToLongBits(out.getDouble(i)));
            }
        }
    }
}
