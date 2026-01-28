package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.ndarray.Shape;
import org.numjvm.exception.ShapeException;
import static org.junit.jupiter.api.Assertions.*;
class MatmulShapeTest {
    @Test void vectorPromotionAndBroadcasting() {
        assertEquals(Shape.of(),MatmulPlan.of(Shape.of(3),Shape.of(3)).output());
        assertEquals(Shape.of(4),MatmulPlan.of(Shape.of(3),Shape.of(3,4)).output());
        assertEquals(Shape.of(2),MatmulPlan.of(Shape.of(2,3),Shape.of(3)).output());
        assertEquals(Shape.of(7,5,2,4),MatmulPlan.of(Shape.of(7,1,2,3),Shape.of(5,3,4)).output());
        assertEquals(Shape.of(0,2,4),MatmulPlan.of(Shape.of(0,2,0),Shape.of(1,0,4)).output());
    }
    @Test void invalidShapesAndOverflowAreRejectedWithoutDataAllocation() {
        assertThrows(ShapeException.class,()->MatmulPlan.of(Shape.of(),Shape.of(1)));
        assertThrows(ShapeException.class,()->MatmulPlan.of(Shape.of(2,3),Shape.of(2,4)));
        assertThrows(ShapeException.class,()->MatmulPlan.of(Shape.of(2,1,1),Shape.of(3,1,1)));
        assertThrows(ShapeException.class,()->MatmulPlan.of(Shape.of(Long.MAX_VALUE,1),Shape.of(1,2)));
    }
}
