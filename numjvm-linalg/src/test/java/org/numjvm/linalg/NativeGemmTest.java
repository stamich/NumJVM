package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
class NativeGemmTest {
    @Test void sgemmRectangularAndPacking() {
        Assumptions.assumeTrue(OpenBlas.INSTANCE!=null,"optional OpenBLAS unavailable");
        var a=NJ.arange(0,6,1).astype(DType.FLOAT32).reshape(3,2).transpose();
        var b=NJ.ones(DType.FLOAT32,3,4);
        try(var pa=PackedMatrix.pack(MatrixView.input(a,true),2,3,DType.FLOAT32);
            var pb=PackedMatrix.pack(MatrixView.input(b,false),3,4,DType.FLOAT32);
            var out=PackedMatrix.allocate(2,4,DType.FLOAT32)) {
            NativeGemm.sgemm(pa.view(),pb.view(),out.view(),2,3,4);
            for(int j=0;j<4;j++) { assertEquals(6,out.view().get(0,j));assertEquals(9,out.view().get(1,j)); }
        }
    }
}
