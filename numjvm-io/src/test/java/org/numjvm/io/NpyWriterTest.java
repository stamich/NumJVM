package org.numjvm.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.MemoryException;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;
import org.numjvm.ndarray.Slice;

class NpyWriterTest {

    @TempDir
    Path dir;

    /**
     * Reading a NumPy file and writing it again reproduces NumPy's bytes exactly.
     */
    @ParameterizedTest
    @ValueSource(strings = {"b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16",
            "b1_fortran", "i8_fortran", "f4_fortran", "c16_fortran", "scalar", "empty", "u8_max", "f2_values"})
    void rewritingNumPyFilesIsByteIdentical(String name) throws IOException {
        Path original = NpyReaderTest.fixture(name + ".npy");
        Path copy = dir.resolve(name + ".npy");
        Npy.write(copy, Npy.read(original));
        assertArrayEquals(Files.readAllBytes(original), Files.readAllBytes(copy), name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"i2", "f8", "c8"})
    void bigEndianFilesAreRewrittenInNativeOrder(String code) throws IOException {
        Path copy = dir.resolve(code + ".npy");
        Npy.write(copy, Npy.read(NpyReaderTest.fixture(code + "_big.npy")));
        assertArrayEquals(Files.readAllBytes(NpyReaderTest.fixture(code + ".npy")), Files.readAllBytes(copy));
    }

    @Test
    void nonContiguousViewsAreWrittenInCOrder() throws IOException {
        NDArray a = NJ.arange(0.0, 24.0, 1.0).reshape(2, 3, 4);
        NDArray[] views = {
                a.permuteDims(1, 0, 2),
                a.slice(Slice.all(), Slice.step(-1), Slice.range(1, 4, 2)),
                NJ.array(new double[]{1, 2, 3}).broadcastTo(2, 3),
        };
        for (NDArray v : views) {
            Path f = dir.resolve("view.npy");
            Npy.write(f, v);
            NDArray back = Npy.read(f);
            assertTrue(back.isCContiguous());
            assertTrue(NJ.arrayEquals(v, back), v.toString());
            assertTrue(new String(Files.readAllBytes(f), StandardCharsets.ISO_8859_1).contains("'fortran_order': False"));
        }
    }

    @Test
    void transposedMatricesAreWrittenAsFortranWithoutCopy() throws IOException {
        NDArray t = NJ.arange(0.0, 6.0, 1.0).reshape(2, 3).transpose();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        NpyWriter.write(out, t);
        String text = out.toString(StandardCharsets.ISO_8859_1);
        assertTrue(text.contains("'fortran_order': True, 'shape': (3, 2)"), text);
        NDArray back = Npy.read(new ByteArrayInputStream(out.toByteArray()));
        assertTrue(back.isFContiguous());
        assertTrue(NJ.arrayEquals(t, back));
    }

    @ParameterizedTest
    @ValueSource(strings = {"BOOL", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64",
            "FLOAT16", "FLOAT32", "FLOAT64"})
    void roundTripEveryRealDType(String name) throws IOException {
        DType t = DType.valueOf(name);
        NDArray a = NJ.array(t, new double[]{0, 1, 2, 3, 4, 5, 6, 7}, 2, 4);
        Path f = dir.resolve(name + ".npy");
        Npy.write(f, a);
        NDArray b = Npy.read(f);
        assertEquals(t, b.dtype());
        assertTrue(NJ.arrayEquals(a, b));
    }

    @Test
    void bfloat16CannotBeSaved() {
        NpyFormatException e = assertThrows(NpyFormatException.class,
                () -> Npy.write(dir.resolve("x.npy"), NJ.zeros(DType.BFLOAT16, 2)));
        assertTrue(e.getMessage().contains("bfloat16"), e.getMessage());
    }

    @Test
    void scopedArraysCanBeWrittenWhileOpenOnly() {
        NDArray a;
        try (NJScope scope = NJ.scope()) {
            a = scope.array(new double[]{1, 2, 3});
            Npy.write(dir.resolve("scoped.npy"), a);
        }
        assertEquals(Shape.of(3), Npy.read(dir.resolve("scoped.npy")).shape());
        NDArray closed = a;
        assertThrows(MemoryException.class, () -> Npy.write(dir.resolve("closed.npy"), closed));
    }

    @Test
    void largeArraysAreWrittenInChunks() throws IOException {
        NDArray big = NJ.arange(0.0, 300_000.0, 1.0);    // 2.4 MB, off-heap
        assertTrue(big.isNative());
        Path f = dir.resolve("big.npy");
        Npy.write(f, big);
        assertEquals(128 + 2_400_000, Files.size(f));
        NDArray back = Npy.read(f);
        assertEquals(299_999.0, back.getDouble(299_999));
        assertFalse(back.sharesMemoryWith(big));
    }
}
