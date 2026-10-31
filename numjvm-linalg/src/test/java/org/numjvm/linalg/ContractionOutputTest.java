package org.numjvm.linalg;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.Slice;
import static org.junit.jupiter.api.Assertions.*;

class ContractionOutputTest {
    @Test void dotProtectsIdenticalOutputView() {
        var a = NJ.array(new double[]{1, 2});
        var b = NJ.array(new double[]{1, 2, 3, 4}, 2, 2);
        assertSame(a, Linalg.dot(a, b, a));
        assertEquals(7, a.getDouble(0));
        assertEquals(10, a.getDouble(1));
    }

    @Test void vecdotProtectsPartiallyOverlappingOutput() {
        var a = NJ.array(new double[]{1, 2, 3, 4}, 2, 2);
        var out = a.slice(Slice.all(), Slice.index(0));
        assertSame(out, Linalg.vecdot(a, NJ.array(new double[]{5, 6}), out));
        assertEquals(17, out.getDouble(0));
        assertEquals(39, out.getDouble(1));
    }

    @Test void invalidOutputIsRejectedBeforeWrites() {
        var out = NJ.ones(2);
        assertThrows(ShapeException.class, () -> Linalg.dot(NJ.ones(2), NJ.ones(2), out));
        assertEquals(1, out.getDouble(0));
        var wrongType = NJ.ones(DType.FLOAT32);
        assertThrows(DTypeException.class, () -> Linalg.dot(NJ.ones(2), NJ.ones(2), wrongType));
        assertEquals(1, wrongType.getDouble());
    }
}
