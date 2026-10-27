package org.numjvm.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Strides;

/**
 * Reads files written by NumPy (see {@code fixtures/generate.py}).
 */
class NpyReaderTest {

    static Path fixture(String name) {
        try {
            return Path.of(NpyReaderTest.class.getResource("fixtures/" + name).toURI());
        } catch (URISyntaxException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * Real and imaginary parts of element k of a complex array, read from raw memory.
     */
    private static double[] complexAt(NDArray a, int i, int j) {
        boolean single = a.dtype() == DType.COMPLEX64;
        NDArray parts = SharedSecrets.arrayAccess().create(SharedSecrets.arrayAccess().buffer(a),
                single ? DType.FLOAT32 : DType.FLOAT64, SharedSecrets.arrayAccess().offset(a),
                Shape.of(2, 3, 2), Strides.of(a.strides().stride(0), a.strides().stride(1), a.itemSize() / 2), false);
        return new double[]{parts.getDouble(i, j, 0), parts.getDouble(i, j, 1)};
    }

    private static void assertArange(NDArray a, String code) {
        assertEquals(Shape.of(2, 3), a.shape(), code);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                int k = 3 * i + j;
                if (a.dtype().isComplex()) {
                    double[] c = complexAt(a, i, j);
                    assertEquals(k, c[0], code);
                    assertEquals(10.0 * k, c[1], code);
                } else if (a.dtype() == DType.BOOL) {
                    assertEquals(k % 2 == 0, a.getBoolean(i, j), code);
                } else {
                    assertEquals(k, a.getDouble(i, j), code);
                }
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16"})
    void readsEverySupportedDTypeInCOrder(String code) {
        NDArray a = Npy.read(fixture(code + ".npy"));
        assertEquals(DType.fromNumpyDescriptor(code), a.dtype());
        assertTrue(a.isCContiguous());
        assertArange(a, code);
    }

    @ParameterizedTest
    @ValueSource(strings = {"b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16"})
    void readsFortranOrderWithoutReordering(String code) {
        NDArray a = Npy.read(fixture(code + "_fortran.npy"));
        assertTrue(a.isFContiguous(), code);
        assertFalse(a.isCContiguous(), code);
        assertArange(a, code);
    }

    @ParameterizedTest
    @ValueSource(strings = {"i2", "i4", "i8", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16"})
    void byteSwapsBigEndianFiles(String code) {
        NDArray a = Npy.read(fixture(code + "_big.npy"));
        assertArange(a, code + " big-endian");
    }

    @Test
    void scalarEmptyAndSpecialValues() {
        NDArray s = Npy.read(fixture("scalar.npy"));
        assertEquals(0, s.ndim());
        assertEquals(2.5, s.getDouble());
        NDArray e = Npy.read(fixture("empty.npy"));
        assertEquals(Shape.of(0, 3), e.shape());
        assertEquals(DType.INT16, e.dtype());
        NDArray u = Npy.read(fixture("u8_max.npy"));
        assertEquals("9223372036854775808", Unsigned.toString(u.getLong(1)));
        assertEquals("18446744073709551615", Unsigned.toString(u.getLong(2)));
        NDArray h = Npy.read(fixture("f2_values.npy"));
        assertEquals(0.0999755859375, h.getDouble(0));
        assertEquals(65504.0, h.getDouble(1));
        assertEquals(Double.NEGATIVE_INFINITY, h.getDouble(2));
        assertTrue(Double.isNaN(h.getDouble(3)));
    }

    @Test
    void readsIntoAScope() {
        NDArray a;
        try (NJScope scope = NJ.scope()) {
            a = Npy.read(fixture("f8.npy"), scope);
            assertSame(scope, a.memoryOwner());
            assertTrue(a.isNative());
            assertEquals(5.0, a.getDouble(1, 2));
        }
        assertFalse(a.isAlive());
    }

    @Test
    void readsFromAStreamAndLeavesItAfterTheData() throws IOException {
        byte[] one = Files.readAllBytes(fixture("i4.npy"));
        byte[] two = Files.readAllBytes(fixture("f8.npy"));
        byte[] both = new byte[one.length + two.length];
        System.arraycopy(one, 0, both, 0, one.length);
        System.arraycopy(two, 0, both, one.length, two.length);
        InputStream in = new ByteArrayInputStream(both);
        assertEquals(DType.INT32, Npy.read(in).dtype());
        assertEquals(DType.FLOAT64, Npy.read(in).dtype());
    }

    @Test
    void truncatedFilesAreDetectedBeforeAllocating(@TempDir Path dir) throws IOException {
        byte[] full = Files.readAllBytes(fixture("f8.npy"));
        Path cut = dir.resolve("cut.npy");
        Files.write(cut, java.util.Arrays.copyOf(full, full.length - 1));
        NpyFormatException e = assertThrows(NpyFormatException.class, () -> Npy.read(cut));
        assertTrue(e.getMessage().contains("truncated"), e.getMessage());
        assertThrows(NpyFormatException.class,
                () -> Npy.read(new ByteArrayInputStream(java.util.Arrays.copyOf(full, full.length - 8))));
    }

    @Test
    void hugeDeclaredShapesFailOnTheSizeCheck(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("huge.npy");
        Files.write(f, NpyHeader.encode("<f8", false, Shape.of(1L << 40)));   // 8 TiB declared, 0 present
        assertThrows(NpyFormatException.class, () -> Npy.read(f));
        assertThrows(NpyFormatException.class, () -> NpyReader.withLimits(NpyLimits.defaults().withMaxDataBytes(1 << 20))
                .read(new ByteArrayInputStream(Files.readAllBytes(f))));
    }

    @Test
    void missingFilesAreIoErrors(@TempDir Path dir) {
        assertThrows(UncheckedIOException.class, () -> Npy.read(dir.resolve("missing.npy")));
    }

    @Test
    void swapBytesReversesEachUnit() {
        byte[] b = {1, 2, 3, 4, 5, 6, 7, 8};
        NpyReader.swapBytes(b, 8, 4);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new byte[]{4, 3, 2, 1, 8, 7, 6, 5}, b);
    }
}
