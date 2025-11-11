package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.exception.DTypeException;
import org.numjvm.exception.IndexException;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.HeapAllocator;
import org.numjvm.memory.NJScope;

class AccessorTest {

    private static final SharedSecrets.ArrayAccess ACCESS = SharedSecrets.arrayAccess();

    private static NDArray zeros(DType dtype, long... shape) {
        return ACCESS.allocate(HeapAllocator.INSTANCE, dtype, Shape.of(shape), Order.C);
    }

    @Test
    void rankSpecialisedAccessorsAddressTheRightElement() {
        NDArray a = zeros(DType.FLOAT64, 2, 3, 4);
        long n = 0;
        for (long i = 0; i < 2; i++) {
            for (long j = 0; j < 3; j++) {
                for (long k = 0; k < 4; k++) {
                    a.setDouble(i, j, k, n++);
                }
            }
        }
        assertEquals(23.0, a.getDouble(1, 2, 3));
        assertEquals(17.0, a.getDouble(new long[]{1, 1, 1}));
        NDArray m = zeros(DType.INT64, 3, 4);
        m.setLong(2, 1, 99);
        assertEquals(99, m.getLong(2, 1));
        assertEquals(99, m.getLong(new long[]{2, 1}));
        NDArray v = zeros(DType.FLOAT32, 5);
        v.setDouble(4, 1.5);
        assertEquals(1.5, v.getDouble(4));
        NDArray s = zeros(DType.INT16);
        s.setLong(-7);
        assertEquals(-7, s.getLong());
        assertEquals(-7.0, s.getDouble(new long[0]));
    }

    @Test
    void indicesAreCheckedPerAxis() {
        NDArray a = zeros(DType.FLOAT64, 2, 3);
        IndexException e = assertThrows(IndexException.class, () -> a.getDouble(1, 3));
        assertEquals("index 3 is out of bounds for axis 1 with size 3", e.getMessage());
        assertThrows(IndexException.class, () -> a.getDouble(2, 0));
        assertThrows(IndexException.class, () -> a.getDouble(-1, 0), "negative indices are not wrapped");
        assertThrows(IndexException.class, () -> a.getDouble(new long[]{0, 0, 0}));
        assertThrows(IndexException.class, () -> a.getDouble(0), "wrong rank");
        assertThrows(IndexException.class, () -> a.getDouble());
        assertThrows(IndexException.class, () -> zeros(DType.FLOAT64, 0).getDouble(0), "empty array");
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"COMPLEX64", "COMPLEX128"}, mode = EnumSource.Mode.EXCLUDE)
    void everyRealDTypeRoundTripsSmallIntegers(DType dtype) {
        NDArray a = zeros(dtype, 3);
        a.setDouble(0, 1.0);
        a.setLong(1, 0);
        a.setBoolean(2, true);
        assertEquals(1.0, a.getDouble(0));
        assertEquals(0.0, a.getDouble(1));
        assertEquals(1.0, a.getDouble(2));
        assertTrue(a.getBoolean(0));
        assertFalse(a.getBoolean(1));
    }

    @ParameterizedTest
    @CsvSource({
            "INT8, 300, 44",           // wraps to 8 bits
            "INT8, -129, 127",
            "UINT8, -1, 255",
            "INT16, 40000, -25536",
            "UINT16, 70000, 4464",
            "INT32, 4294967297, 1",
            "UINT32, -1, 4294967295",
            "BOOL, 5, 1",
    })
    void setLongNarrowsLikeNumPyCasting(DType dtype, long value, double expected) {
        NDArray a = zeros(dtype, 1);
        a.setLong(0, value);
        assertEquals(expected, a.getDouble(0));
    }

    @ParameterizedTest
    @CsvSource({
            "INT32, 2.9, 2",
            "INT32, -2.9, -2",
            "INT64, 1e18, 1000000000000000000",
            "UINT8, 255.7, 255",
            "BOOL, 0.0, 0",
            "BOOL, -0.5, 1",
            "INT64, NaN, 0",
    })
    void setDoubleTruncatesTowardZero(DType dtype, double value, long expected) {
        NDArray a = zeros(dtype, 1);
        a.setDouble(0, value);
        assertEquals(expected, a.getLong(0));
    }

    @Test
    void uint64AboveTwoToThe63() {
        NDArray a = zeros(DType.UINT64, 3);
        a.setLong(0, -1L);                       // 2^64 - 1
        a.setDouble(1, 0x1p63);                   // exactly 2^63
        a.setDouble(2, 1.5e19);
        assertEquals(-1L, a.getLong(0), "getLong returns the raw bits");
        assertEquals("18446744073709551615", Unsigned.toString(a.getLong(0)));
        assertEquals(0x1p64, a.getDouble(0));
        assertEquals(Long.MIN_VALUE, a.getLong(1));
        assertEquals("9223372036854775808", Unsigned.toString(a.getLong(1)));
        assertEquals(0x1p63, a.getDouble(1));
        assertEquals("15000000000000000000", Unsigned.toString(a.getLong(2)));
        assertEquals(1.5e19, a.getDouble(2));
    }

    @Test
    void unsignedHelpers() {
        assertEquals(0x1p63, Unsigned.toDouble(Long.MIN_VALUE));
        assertEquals(42.0, Unsigned.toDouble(42));
        assertEquals(-1L, Unsigned.fromDouble(0x1p64));
        assertEquals(-1L, Unsigned.fromDouble(-1.0), "negative values wrap");
        assertEquals(0L, Unsigned.fromDouble(Double.NaN));
        // correct rounding at the top of the range: 2^64 - 1 rounds to 2^64
        assertEquals(0x1p64, Unsigned.toDouble(-1L));
        assertEquals(1L << 62, Unsigned.fromDouble(0x1p62));
    }

    @Test
    void halfPrecisionTypes() {
        NDArray h = zeros(DType.FLOAT16, 3);
        h.setDouble(0, 0.1);
        h.setDouble(1, 65504.0);
        h.setDouble(2, 1e6);
        assertEquals(0.0999755859375, h.getDouble(0), "nearest binary16 to 0.1");
        assertEquals(65504.0, h.getDouble(1), "largest finite binary16");
        assertEquals(Double.POSITIVE_INFINITY, h.getDouble(2), "overflow");

        NDArray b = zeros(DType.BFLOAT16, 4);
        b.setDouble(0, 1.0);
        b.setDouble(1, 3.14159);
        b.setDouble(2, Double.NaN);
        b.setDouble(3, 1.00390625);              // halfway between 1.0 and 1.0078125: ties to even
        assertEquals(1.0, b.getDouble(0));
        assertEquals(3.140625, b.getDouble(1));
        assertTrue(Double.isNaN(b.getDouble(2)));
        assertEquals(1.0, b.getDouble(3));
    }

    @Test
    void getLongIsNotDefinedForFloats() {
        assertThrows(DTypeException.class, () -> zeros(DType.FLOAT64, 1).getLong(0));
    }

    @Test
    void getBooleanTreatsNaNAsTrue() {
        NDArray a = zeros(DType.FLOAT32, 2);
        a.setDouble(1, Double.NaN);
        assertFalse(a.getBoolean(0));
        assertTrue(a.getBoolean(1));
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"COMPLEX64", "COMPLEX128"})
    void complexIsStorageOnly(DType dtype) {
        NDArray a = zeros(dtype, 2);
        assertThrows(DTypeException.class, () -> a.getDouble(0));
        assertThrows(DTypeException.class, () -> a.getLong(0));
        assertThrows(DTypeException.class, () -> a.getBoolean(0));
        assertThrows(DTypeException.class, () -> a.setDouble(0, 1));
        assertThrows(DTypeException.class, () -> a.setLong(0, 1));
    }

    @Test
    void readOnlyArraysRejectWrites() {
        NDArray ro = ACCESS.create(HeapAllocator.INSTANCE.allocate(24), DType.FLOAT64, 0, Shape.of(4, 3),
                Strides.of(0, 8), false);
        ReadOnlyArrayException e = assertThrows(ReadOnlyArrayException.class, () -> ro.setDouble(0, 0, 1.0));
        assertTrue(e.getMessage().contains("broadcast"), e.getMessage());
        assertThrows(ReadOnlyArrayException.class, () -> ro.setLong(new long[]{1, 1}, 1));
        assertThrows(ReadOnlyArrayException.class, () -> ro.setBoolean(3, 2, true));
        assertEquals(0.0, ro.getDouble(3, 2));
    }

    @Test
    void closedScopeAccessIsAMemoryException() {
        NDArray a;
        try (NJScope scope = NJ.scope()) {
            a = ACCESS.allocate(SharedSecrets.scopeAccess().allocator(scope), DType.FLOAT64, Shape.of(4), Order.C);
            a.setDouble(0, 1.0);
        }
        MemoryException e = assertThrows(MemoryException.class, () -> a.getDouble(0));
        assertTrue(e.getMessage().contains("closed"), e.getMessage());
        assertThrows(MemoryException.class, () -> a.setDouble(0, 2.0));
    }

    @Test
    void confinedScopeAccessFromAnotherThreadIsAMemoryException() throws InterruptedException {
        try (NJScope scope = NJ.scope()) {
            NDArray a = ACCESS.allocate(SharedSecrets.scopeAccess().allocator(scope), DType.INT32, Shape.of(4),
                    Order.C);
            ExecutionException e = assertThrows(ExecutionException.class,
                    () -> CompletableFuture.supplyAsync(() -> a.getLong(0)).get());
            assertTrue(e.getCause() instanceof MemoryException, e.getCause().toString());
        }
    }

    @Test
    void rankSpecialisedAccessDoesNotAllocate() {
        NDArray a = zeros(DType.FLOAT64, 64, 64);
        com.sun.management.ThreadMXBean threads =
                (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        long tid = Thread.currentThread().threadId();
        double sink = 0;
        for (int warm = 0; warm < 3; warm++) {
            sink += sumAll(a);
        }
        long before = threads.getThreadAllocatedBytes(tid);
        for (int rep = 0; rep < 50; rep++) {
            sink += sumAll(a);
        }
        long allocated = threads.getThreadAllocatedBytes(tid) - before;
        // 50 x 4,096 accesses; even a single 16-byte allocation per access would be 3 MB.
        assertTrue(allocated < 64 * 1024, "allocated " + allocated + " bytes during 204,800 accesses");
        assertEquals(0.0, sink);
    }

    private static double sumAll(NDArray a) {
        double s = 0;
        for (long i = 0; i < 64; i++) {
            for (long j = 0; j < 64; j++) {
                s += a.getDouble(i, j);
            }
        }
        return s;
    }
}
