package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;
class MatrixPackingTest {
    @Test void packingConvertsNegativeStridesAndReleasesMemory() {
        var a=NJ.arange(0,12,1).astype(DType.FLOAT32).reshape(3,4).slice(Slice.step(-1),Slice.step(-1));
        MatrixView view;
        try(var packed=PackedMatrix.pack(MatrixView.input(a,true),3,4,DType.FLOAT64)) {
            view=packed.view();
            for(int i=0;i<3;i++)for(int j=0;j<4;j++)assertEquals(a.getDouble(i,j),view.get(i,j));
        }
        assertFalse(view.memory().scope().isAlive());
    }
    @Test void aliasingAndStridedOutputsUseCopyback() {
        var a=NJ.array(new double[]{1,2,3,4},2,2);var b=NJ.array(new double[]{2,0,0,2},2,2);
        Linalg.matmul(a,b,a);assertEquals(2,a.getDouble(0,0));assertEquals(8,a.getDouble(1,1));
        var out=NJ.zeros(2,4).slice(Slice.all(),Slice.step(2));
        Linalg.matmul(a,b,out);assertEquals(4,out.getDouble(0,0));assertEquals(16,out.getDouble(1,1));
    }
}
