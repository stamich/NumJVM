package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.ndarray.Shape;
import static org.junit.jupiter.api.Assertions.*;
class BatchMatmulTest {
    @Test void broadcastsBatchesAndVectorPromotions() {
        var a=NJ.ones(2,1,3,4);var b=NJ.ones(5,4,6);
        var c=Linalg.matmul(a,b);assertEquals(Shape.of(2,5,3,6),c.shape());
        var indices=new org.numjvm.iteration.IndexIterator(c.shape());
        while(indices.hasNext())assertEquals(4,c.getDouble(indices.next()));
        var v=Linalg.matmul(NJ.ones(4),b);assertEquals(Shape.of(5,6),v.shape());assertEquals(4,v.getDouble(4,5));
        var empty=Linalg.matmul(NJ.ones(0,3,4),NJ.ones(1,4,2));assertEquals(Shape.of(0,3,2),empty.shape());
    }
}
