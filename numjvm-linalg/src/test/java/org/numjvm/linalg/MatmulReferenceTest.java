package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
class MatmulReferenceTest {
    @Test void vectorMatrixAndMixedDtypes() {
        var a=NJ.array(new double[]{1,2,3,4,5,6},2,3);
        var b=NJ.array(new double[]{1,2,3});
        var actual=Linalg.matmul(a,b);
        assertEquals(14,actual.getDouble(0));assertEquals(32,actual.getDouble(1));
        assertEquals(14,Linalg.matmul(b,b).getDouble());
        assertEquals(14,Linalg.matmul(b.astype(DType.FLOAT32),b).getDouble());
        assertEquals(DType.FLOAT32,Linalg.matmul(b.astype(DType.FLOAT32),b.astype(DType.FLOAT32)).dtype());
    }
    @Test void zeroContractionOverwritesOldOutput() {
        var out=NJ.ones(2,3);Linalg.matmul(NJ.zeros(2,0),NJ.zeros(0,3),out);
        for(int i=0;i<2;i++)for(int j=0;j<3;j++)assertEquals(0,out.getDouble(i,j));
    }
}
