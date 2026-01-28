package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
class BlockedMatmulTest {
    @Test void rectangularAndIncompleteBlocksPreserveSequentialContractions() {
        for(DType t:new DType[]{DType.FLOAT32,DType.FLOAT64}) {
            var a=NJ.arange(0,37*41,1).astype(t).reshape(37,41);
            var b=NJ.arange(0,41*39,1).astype(t).reshape(41,39);
            var expected=NJ.zeros(t,37,39);
            JavaMatmul.reference(MatrixView.input(a,true),MatrixView.input(b,false),MatrixView.output(expected,MatmulPlan.of(a.shape(),b.shape())),37,41,39);
            var actual=Linalg.matmul(a,b);
            for(int i=0;i<37;i++)for(int j=0;j<39;j++)assertEquals(expected.getDouble(i,j),actual.getDouble(i,j));
        }
    }
}
