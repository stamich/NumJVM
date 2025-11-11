package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.MemoryException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.AutomaticOwner;
import org.numjvm.memory.NJScope;

class CopyTest {

    private static final SharedSecrets.ArrayAccess ACCESS = SharedSecrets.arrayAccess();

    /** A 2x3 float64 array [[0,1,2],[3,4,5]] and views of it built from raw metadata. */
    private static NDArray base() {
        return NJ.array(new double[]{0, 1, 2, 3, 4, 5}, 2, 3);
    }

    private static NDArray view(NDArray a, long offset, long[] shape, long[] strides, boolean writeable) {
        return ACCESS.create(ACCESS.buffer(a), a.dtype(), offset, Shape.of(shape), Strides.of(strides), writeable);
    }

    private static double[] values(NDArray a) {
        NDArray c = a.copy();
        double[] out = new double[(int) c.size()];
        NDArray flat = view(c, 0, new long[]{c.size()}, new long[]{c.itemSize()}, false);
        for (int i = 0; i < out.length; i++) {
            out[i] = flat.getDouble(i);
        }
        return out;
    }

    @Test
    void copyOfAContiguousArray() {
        NDArray a = base();
        NDArray c = a.copy();
        assertNotSame(a, c);
        assertTrue(c.isCContiguous());
        assertEquals(a.shape(), c.shape());
        c.setDouble(0, 0, 99);
        assertEquals(0.0, a.getDouble(0, 0), "the copy has its own memory");
    }

    @Test
    void copiesOfViewsHaveTheViewsLogicalValues() {
        NDArray a = base();
        NDArray t = view(a, 0, new long[]{3, 2}, new long[]{8, 24}, true);
        assertArrayEquals(new double[]{0, 3, 1, 4, 2, 5}, values(t));
        NDArray reversed = view(a, 40, new long[]{6}, new long[]{-8}, true);
        assertArrayEquals(new double[]{5, 4, 3, 2, 1, 0}, values(reversed));
        NDArray everyOther = view(a, 0, new long[]{2, 2}, new long[]{24, 16}, true);
        assertArrayEquals(new double[]{0, 2, 3, 5}, values(everyOther));
        NDArray row = NJ.array(new double[]{7, 8, 9});
        NDArray broadcast = view(row, 0, new long[]{2, 3}, new long[]{0, 8}, false);
        NDArray bc = broadcast.copy();
        assertArrayEquals(new double[]{7, 8, 9, 7, 8, 9}, values(bc));
        assertTrue(bc.isWriteable(), "copies of read-only views are writeable");
        assertEquals(Strides.of(24, 8), bc.strides());
    }

    @Test
    void fortranOrderCopy() {
        NDArray f = base().copy(Order.F);
        assertTrue(f.isFContiguous());
        assertFalse(f.isCContiguous());
        assertEquals(Strides.of(8, 16), f.strides());
        assertEquals(5.0, f.getDouble(1, 2));
        NDArray back = f.copy(Order.C);
        assertArrayEquals(new double[]{0, 1, 2, 3, 4, 5}, values(back));
        NDArray fOfF = f.copy(Order.F);
        assertEquals(4.0, fOfF.getDouble(1, 1));
    }

    @Test
    void scalarAndEmptyCopies() {
        assertEquals(2.5, NJ.scalar(2.5).copy().getDouble());
        NDArray empty = NJ.zeros(DType.INT32, 0, 3);
        assertEquals(Shape.of(0, 3), empty.copy().shape());
        assertEquals(Shape.of(0, 3), empty.copy(Order.F).shape());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void everyDTypeCanBeCopiedBitForBit(DType dtype) {
        NDArray a = NJ.zeros(dtype, 4);
        NDArray bytes = ACCESS.create(ACCESS.buffer(a), DType.UINT8, 0, Shape.of(a.nbytes()), Strides.of(1), true);
        for (long i = 0; i < a.nbytes(); i++) {
            bytes.setLong(i, i * 7 + 1);
        }
        NDArray reversed = view(a, 3L * dtype.size(), new long[]{4}, new long[]{-dtype.size()}, true);
        NDArray c = reversed.copy();
        NDArray cb = ACCESS.create(ACCESS.buffer(c), DType.UINT8, 0, Shape.of(c.nbytes()), Strides.of(1), true);
        for (int e = 0; e < 4; e++) {
            for (int b = 0; b < dtype.size(); b++) {
                long srcByte = (3L - e) * dtype.size() + b;
                assertEquals((srcByte * 7 + 1) & 0xFF, cb.getLong((long) e * dtype.size() + b));
            }
        }
    }

    @ParameterizedTest
    @CsvSource({
            "FLOAT64, INT32, 2.9, 2",
            "FLOAT64, UINT8, 300.0, 44",
            "FLOAT64, BOOL, 0.5, 1",
            "FLOAT64, FLOAT16, 0.1, 0.0999755859375",
            "FLOAT64, BFLOAT16, 3.14159, 3.140625",
            "INT64, FLOAT32, 16777217, 16777216",
            "BOOL, FLOAT64, 1, 1",
            "FLOAT32, FLOAT64, 0.1, 0.10000000149011612",
    })
    void astypeConvertsLikeNumPy(DType from, DType to, double value, double expected) {
        NDArray a = NJ.array(from, new double[]{value});
        NDArray b = a.astype(to);
        assertEquals(to, b.dtype());
        assertEquals(expected, b.getDouble(0));
    }

    @Test
    void integerConversionsKeepAll64Bits() {
        NDArray a = NJ.array(new long[]{Long.MAX_VALUE, -1});
        NDArray u = a.astype(DType.UINT64);
        assertEquals(Long.MAX_VALUE, u.getLong(0));
        assertEquals(-1L, u.getLong(1), "-1 becomes 2^64 - 1 (raw bits preserved)");
        assertEquals(Long.MAX_VALUE, u.astype(DType.INT64).getLong(0));
        assertEquals(-1L, NJ.array(new long[]{255}).astype(DType.INT8).getLong(0));
    }

    @Test
    void complexConversions() {
        NDArray real = NJ.array(new double[]{1.5, -2});
        NDArray c = real.astype(DType.COMPLEX128);
        NDArray parts = ACCESS.create(ACCESS.buffer(c), DType.FLOAT64, 0, Shape.of(4), Strides.of(8), false);
        assertArrayEquals(new double[]{1.5, 0, -2, 0}, values(parts));
        NDArray c64 = c.astype(DType.COMPLEX64);
        NDArray p64 = ACCESS.create(ACCESS.buffer(c64), DType.FLOAT32, 0, Shape.of(4), Strides.of(4), false);
        assertArrayEquals(new double[]{1.5, 0, -2, 0}, values(p64));
        DTypeException e = assertThrows(DTypeException.class, () -> c.astype(DType.FLOAT64));
        assertTrue(e.getMessage().contains("imaginary"), e.getMessage());
    }

    @Test
    void astypeToTheSameDTypeIsAPlainCopy() {
        NDArray a = base();
        NDArray b = a.astype(DType.FLOAT64);
        b.setDouble(0, 0, 9);
        assertEquals(0.0, a.getDouble(0, 0));
        assertTrue(a.astype(DType.FLOAT64, Order.F).isFContiguous());
    }

    @Test
    void keepCopiesOutOfTheScope() {
        NDArray kept;
        NDArray keptView;
        NDArray scoped;
        try (NJScope scope = NJ.scope()) {
            scoped = scope.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
            kept = scope.keep(scoped);
            keptView = scope.keep(view(scoped, 0, new long[]{3, 2}, new long[]{8, 24}, false));
            assertEquals(AutomaticOwner.INSTANCE, kept.memoryOwner());
        }
        assertFalse(scoped.isAlive());
        assertThrows(MemoryException.class, () -> scoped.getDouble(0, 0));
        assertEquals(6.0, kept.getDouble(1, 2));
        assertEquals(Strides.of(24, 8), kept.strides());
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, values(keptView));
        assertTrue(keptView.isWriteable());
    }

    @Test
    void keepAfterCloseFails() {
        NJScope scope = NJ.scope();
        NDArray a = scope.zeros(DType.FLOAT64, 2);
        scope.close();
        assertThrows(MemoryException.class, () -> scope.keep(a));
    }

    @Test
    void copyingAClosedScopesArrayFails() {
        NDArray a;
        try (NJScope scope = NJ.scope()) {
            a = scope.zeros(DType.FLOAT64, 2, 2);
        }
        assertThrows(MemoryException.class, a::copy);
    }

    private static void assertArrayEquals(double[] expected, double[] actual) {
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, actual);
    }
}
