package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.*;
import static org.junit.jupiter.api.Assertions.*;
class MatmulMemoryTest {
    @Test void exactOutputDtypeAndScopeRules() {
        var a=NJ.zeros(DType.FLOAT32,2,3);var b=NJ.zeros(3,4);
        assertEquals(DType.FLOAT64,MatmulMemory.dtype(a,b));
        assertThrows(DTypeException.class,()->MatmulMemory.dtype(NJ.zeros(DType.INT32,1),b));
        var p=MatmulPlan.of(a.shape(),b.shape());
        assertThrows(DTypeException.class,()->MatmulMemory.output(NJ.zeros(DType.FLOAT32,2,4),p,DType.FLOAT64));
        assertThrows(ShapeException.class,()->MatmulMemory.output(NJ.zeros(4,2),p,DType.FLOAT64));
        try(var scope=NJ.scope()) {
            var scoped=scope.zeros(DType.FLOAT64,2,3);
            var result=MatmulMemory.allocate(scoped,b,p,DType.FLOAT64);
            assertSame(scope,result.memoryOwner());
        }
        var scope=NJ.scope();var closed=scope.zeros(DType.FLOAT64,0);scope.close();
        assertThrows(MemoryException.class,()->MatmulMemory.accessible(closed));
    }
}
