package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.HeapAllocator;
import org.numjvm.internal.memory.NativeAllocator;

/**
 * Expected flags are those NumPy 2.x reports for the same layouts, e.g.
 * {@code np.arange(6.).reshape(2, 3)[:, ::2].flags}.
 */
class ArrayFlagsTest {

    private static final SharedSecrets.ArrayAccess ACCESS = SharedSecrets.arrayAccess();

    private static NDArray view(long bufferBytes, DType dtype, long offset, long[] shape, long[] strides,
                                boolean writeable) {
        return ACCESS.create(HeapAllocator.INSTANCE.allocate(bufferBytes), dtype, offset, Shape.of(shape),
                Strides.of(strides), writeable);
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            // name; shape; strides; offset; C; F
            "contiguous 2x3;     2,3;   24,8;  0;  true;  false",
            "transposed;         3,2;   8,24;  0;  false; true",
            "column a[:, 1:2];   2,1;   24,8;  8;  false; false",
            "row a[1];           3;     8;     24; true;  true",
            "every other column; 2,2;   24,16; 0;  false; false",
            "reversed;           6;     -8;    40; false; false",
            "length-1 axes;      1,3,1; 999,8,-5; 0; true; true",
            "empty;              0,3;   -8,99; 0;  true;  true",
            "scalar;             '';    '';    16; true;  true",
    })
    void contiguityMatchesNumPy(String name, String shape, String strides, long offset, boolean c, boolean f) {
        NDArray a = view(48, DType.FLOAT64, offset, parse(shape), parse(strides), true);
        assertEquals(c, a.flags().isCContiguous(), name + " C");
        assertEquals(f, a.flags().isFContiguous(), name + " F");
        assertEquals(c, a.isCContiguous());
        assertEquals(f, a.isFContiguous());
        assertEquals(c || f, a.flags().isContiguous());
    }

    private static long[] parse(String csv) {
        if (csv == null || csv.isBlank()) {
            return new long[0];
        }
        return java.util.Arrays.stream(csv.split(",")).mapToLong(Long::parseLong).toArray();
    }

    @Test
    void writeableFlag() {
        assertTrue(view(48, DType.FLOAT64, 0, new long[]{6}, new long[]{8}, true).flags().isWriteable());
        NDArray ro = view(48, DType.FLOAT64, 0, new long[]{4, 3}, new long[]{0, 8}, false);
        assertFalse(ro.flags().isWriteable());
        assertFalse(ro.isWriteable());
    }

    @Test
    void alignment() {
        assertTrue(view(48, DType.FLOAT64, 8, new long[]{2}, new long[]{16}, true).flags().isAligned());
        assertFalse(view(48, DType.INT32, 2, new long[]{2}, new long[]{4}, true).flags().isAligned(), "offset 2");
        assertFalse(view(48, DType.INT32, 0, new long[]{2}, new long[]{6}, true).flags().isAligned(), "stride 6");
        assertTrue(view(48, DType.INT32, 0, new long[]{1}, new long[]{6}, true).flags().isAligned(),
                "the stride of a length-1 axis is irrelevant");
        assertTrue(view(48, DType.COMPLEX128, 8, new long[]{2}, new long[]{16}, true).flags().isAligned(),
                "complex128 needs 8-byte alignment, not 16");
        NDArray nat = ACCESS.create(NativeAllocator.automatic().allocate(64), DType.FLOAT64, 0, Shape.of(8),
                Strides.of(8), true);
        assertTrue(nat.flags().isAligned());
    }

    @Test
    void numpyStyleToString() {
        NDArray a = view(48, DType.FLOAT64, 0, new long[]{2, 3}, new long[]{24, 8}, true);
        assertEquals("  C_CONTIGUOUS : True\n  F_CONTIGUOUS : False\n  WRITEABLE : True\n  ALIGNED : True",
                a.flags().toString());
        assertEquals(ArrayFlags.C_CONTIGUOUS | ArrayFlags.WRITEABLE | ArrayFlags.ALIGNED, a.flags().bits());
        assertEquals(a.flags(), a.flags());
    }
}
