package org.numjvm.exception;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.List;

import org.junit.jupiter.api.Test;

class ExceptionHierarchyTest {

    @Test
    void allExceptionsAreUncheckedNumJVMExceptions() {
        Throwable cause = new ArithmeticException("overflow");
        List<RuntimeException> all = List.of(
                new NumJVMException("x"),
                new ShapeException("x"),
                new ShapeException("x", cause),
                new DTypeException("x"),
                new IndexException("x"),
                new ReadOnlyArrayException("x"),
                new MemoryException("x"),
                new MemoryException("x", cause));
        for (RuntimeException e : all) {
            assertInstanceOf(NumJVMException.class, e);
            assertEquals("x", e.getMessage());
        }
        assertSame(cause, new ShapeException("x", cause).getCause());
    }

    @Test
    void indexOutOfBoundsMessageNamesAxisAndSize() {
        assertEquals("index 7 is out of bounds for axis 1 with size 3",
                IndexException.outOfBounds(7, 1, 3).getMessage());
    }
}
