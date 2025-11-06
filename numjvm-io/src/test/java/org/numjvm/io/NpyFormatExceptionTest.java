package org.numjvm.io;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;
import org.numjvm.exception.NumJVMException;

class NpyFormatExceptionTest {

    @Test
    void isPartOfTheNumJVMHierarchy() {
        Exception cause = new java.io.EOFException();
        NpyFormatException e = new NpyFormatException("truncated", cause);
        assertInstanceOf(NumJVMException.class, e);
        assertSame(cause, e.getCause());
    }
}
