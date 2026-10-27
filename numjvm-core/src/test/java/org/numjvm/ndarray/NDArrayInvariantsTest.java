package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.AutomaticOwner;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.internal.memory.HeapAllocator;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.memory.NJScope;

class NDArrayInvariantsTest {

    private static final SharedSecrets.ArrayAccess ACCESS = SharedSecrets.arrayAccess();

    private static Buffer bytes(long n) {
        return HeapAllocator.INSTANCE.allocate(n);
    }

    private static NDArray create(Buffer b, long offset, long[] shape, long[] strides) {
        return ACCESS.create(b, DType.FLOAT64, offset, Shape.of(shape), Strides.of(strides), false);
    }

    @Test
    void allocatedArrayHasContiguousMetadata() {
        NDArray a = ACCESS.allocate(HeapAllocator.INSTANCE, DType.FLOAT64, Shape.of(2, 3), Order.C);
        assertEquals(DType.FLOAT64, a.dtype());
        assertEquals(Shape.of(2, 3), a.shape());
        assertEquals(Strides.of(24, 8), a.strides());
        assertEquals(2, a.ndim());
        assertEquals(6, a.size());
        assertEquals(3, a.dim(-1));
        assertEquals(8, a.itemSize());
        assertEquals(48, a.nbytes());
        assertTrue(a.isWriteable());
        assertTrue(a.isAlive());
        assertFalse(a.isNative());
        assertSame(AutomaticOwner.INSTANCE, a.memoryOwner());
        NDArray f = ACCESS.allocate(HeapAllocator.INSTANCE, DType.INT32, Shape.of(2, 3), Order.F);
        assertEquals(Strides.of(4, 8), f.strides());
    }

    @Test
    void validViewsAreAccepted() {
        Buffer b = bytes(48);
        assertDoesNotThrow(() -> create(b, 0, new long[]{2, 3}, new long[]{24, 8}));
        assertDoesNotThrow(() -> create(b, 0, new long[]{3, 2}, new long[]{8, 24}), "transposed");
        assertDoesNotThrow(() -> create(b, 40, new long[]{6}, new long[]{-8}), "reversed");
        assertDoesNotThrow(() -> create(b, 0, new long[]{4, 3}, new long[]{0, 8}), "broadcast");
        assertDoesNotThrow(() -> create(b, 40, new long[]{}, new long[]{}), "0-D at the last element");
    }

    @Test
    void viewsReachingOutsideTheBufferAreRejected() {
        Buffer b = bytes(48);
        assertThrows(ShapeException.class, () -> create(b, 8, new long[]{2, 3}, new long[]{24, 8}), "one past the end");
        assertThrows(ShapeException.class, () -> create(b, 0, new long[]{6}, new long[]{-8}), "reversed from 0");
        assertThrows(ShapeException.class, () -> create(b, -8, new long[]{1}, new long[]{8}), "negative offset");
        assertThrows(ShapeException.class, () -> create(b, 48, new long[]{}, new long[]{}), "0-D past the end");
        assertThrows(ShapeException.class, () -> create(b, 0, new long[]{2}, new long[]{Long.MAX_VALUE}), "overflow");
    }

    @Test
    void emptyArraysMayHaveAnyStridesAndOffset() {
        Buffer b = bytes(0);
        assertDoesNotThrow(() -> create(b, 1000, new long[]{0, 5}, new long[]{-99, 12345}));
        assertEquals(0, create(b, 0, new long[]{0}, new long[]{8}).size());
    }

    @Test
    void rankMismatchIsRejected() {
        assertThrows(ShapeException.class, () -> create(bytes(48), 0, new long[]{2, 3}, new long[]{8}));
    }

    @Test
    void nullDTypeIsRejected() {
        assertThrows(NullPointerException.class,
                () -> ACCESS.create(bytes(8), null, 0, Shape.of(1), Strides.of(8), true));
    }

    @Test
    void writeableBroadcastViewsAreRejected() {
        Buffer b = bytes(24);
        assertThrows(ShapeException.class,
                () -> ACCESS.create(b, DType.FLOAT64, 0, Shape.of(4, 3), Strides.of(0, 8), true));
        // a zero stride on an axis of length 1 is harmless
        assertDoesNotThrow(() -> ACCESS.create(b, DType.FLOAT64, 0, Shape.of(1, 3), Strides.of(0, 8), true));
    }

    @Test
    void arraysCannotBeCreatedOnClosedMemory() {
        NJScope scope = NJ.scope();
        Buffer b = SharedSecrets.scopeAccess().allocator(scope).allocate(64);
        scope.close();
        assertThrows(MemoryException.class, () -> create(b, 0, new long[]{8}, new long[]{8}));
    }

    @Test
    void scopedArraysFollowTheirScope() {
        NJScope scope = NJ.scope();
        NDArray a = ACCESS.allocate(SharedSecrets.scopeAccess().allocator(scope), DType.FLOAT32, Shape.of(10), Order.C);
        assertTrue(a.isNative());
        assertSame(scope, a.memoryOwner());
        assertTrue(a.isAlive());
        scope.close();
        assertFalse(a.isAlive());
    }

    @Test
    void sizeOverflowInAllocationIsRejected() {
        assertThrows(ShapeException.class, () -> ACCESS.allocate(HybridAllocator.defaultInstance(),
                DType.COMPLEX128, Shape.of(1L << 60), Order.C));
    }
}
