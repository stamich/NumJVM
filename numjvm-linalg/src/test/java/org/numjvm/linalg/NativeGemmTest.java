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
    @Test void dgemmRectangularAndPacking() {
        Assumptions.assumeTrue(OpenBlas.INSTANCE!=null,"optional OpenBLAS unavailable");
        var a=NJ.arange(0,6,1).astype(DType.FLOAT64).reshape(3,2).transpose();
        var b=NJ.ones(DType.FLOAT64,3,4);
        try(var pa=PackedMatrix.pack(MatrixView.input(a,true),2,3,DType.FLOAT64);
            var pb=PackedMatrix.pack(MatrixView.input(b,false),3,4,DType.FLOAT64);
            var out=PackedMatrix.allocate(2,4,DType.FLOAT64)) {
            NativeGemm.dgemm(pa.view(),pb.view(),out.view(),2,3,4);
            for(int j=0;j<4;j++) { assertEquals(6,out.view().get(0,j));assertEquals(9,out.view().get(1,j)); }
        }
    }
    @Test void rejectsDimensionTruncationAndHeapAddresses() {
        Assumptions.assumeTrue(OpenBlas.INSTANCE!=null,"optional OpenBLAS unavailable");
        assertThrows(IllegalArgumentException.class,()->NativeGemm.dimension((long)Integer.MAX_VALUE+1));
        assertThrows(IllegalArgumentException.class,()->NativeGemm.dimension(-1));
        var a=NJ.ones(2,2);var view=MatrixView.input(a,true);
        assertThrows(IllegalArgumentException.class,()->NativeGemm.dgemm(view,view,view,2,2,2));
    }
}
