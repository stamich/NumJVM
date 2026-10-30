package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import java.util.concurrent.CompletableFuture;
import static org.junit.jupiter.api.Assertions.*;
class MatmulLifetimeTest {
    @Test void resultBelongsToFirstScopeAndOutCanOutliveInputs() {
        var out=NJ.zeros(2,2);org.numjvm.ndarray.NDArray result;
        try(var scope=NJ.scope()) {
            var a=scope.ones(DType.FLOAT64,2,2);
            result=Linalg.matmul(a,NJ.ones(2,2));
            assertSame(scope,result.memoryOwner());
            Linalg.matmul(a,NJ.ones(2,2),out);
        }
        assertFalse(result.isAlive());assertEquals(2,out.getDouble(0,0));
    }
    @Test void wrongThreadIsRejectedBeforeOutputChangesEvenForEmptyInputs() {
        try(var scope=NJ.scope()) {
            var a=scope.zeros(DType.FLOAT64,2,0);var b=NJ.zeros(0,2);var out=NJ.ones(2,2);
            CompletableFuture.runAsync(()->assertThrows(MemoryException.class,()->Linalg.matmul(a,b,out))).join();
            assertEquals(1,out.getDouble(0,0));
        }
    }
    @Test void repeatedPackingReleasesNativeMemoryDeterministically() {
        long before=NJ.memoryStats().liveNativeBytes();
        var a=NJ.ones(3,4);
        for(int i=0;i<100;i++)try(var pack=PackedMatrix.pack(MatrixView.input(a,true),3,4,DType.FLOAT64)) {
            assertEquals(1,pack.view().get(0,0));
        }
        assertEquals(before,NJ.memoryStats().liveNativeBytes());
    }
}
