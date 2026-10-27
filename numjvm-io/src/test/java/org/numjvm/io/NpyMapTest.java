package org.numjvm.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.numjvm.NJ;
import org.numjvm.exception.MemoryException;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.kernels.ValidationKernels;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Shape;

class NpyMapTest {

    @Test
    void mapsWithoutCopyingAndEqualsRead() {
        Path f = NpyReaderTest.fixture("f8.npy");
        try (NJScope scope = NJ.scope()) {
            NDArray m = Npy.map(f, scope);
            assertTrue(m.isNative());
            assertSame(scope, m.memoryOwner());
            assertFalse(m.isWriteable());
            assertTrue(m.flags().isAligned());
            assertTrue(NJ.arrayEquals(Npy.read(f), m));
            assertThrows(ReadOnlyArrayException.class, () -> m.setDouble(0, 0, 1.0));
        }
    }

    @Test
    void fortranFilesMapAsFortranArrays() {
        try (NJScope scope = NJ.scope()) {
            NDArray m = Npy.map(NpyReaderTest.fixture("c16_fortran.npy"), scope);
            assertTrue(m.isFContiguous());
            assertEquals(Shape.of(2, 3), m.shape());
        }
    }

    @Test
    void mappingEndsWithTheScope() {
        NDArray m;
        try (NJScope scope = NJ.scope()) {
            m = Npy.map(NpyReaderTest.fixture("i4.npy"), scope);
            assertEquals(5, m.getLong(1, 2));
        }
        assertFalse(m.isAlive());
        assertThrows(MemoryException.class, () -> m.getLong(0, 0));
    }

    @Test
    void largeFilesAreMappedLazily(@TempDir Path dir) {
        Path f = dir.resolve("big.npy");
        Npy.write(f, NJ.arange(0.0, 1_000_000.0, 1.0));
        try (NJScope scope = NJ.scope()) {
            NDArray m = Npy.map(f, scope);
            assertEquals(999_999.0, m.getDouble(999_999));
            assertEquals(499_999_500_000.0, ValidationKernels.sumFloat64(m));
        }
    }

    @Test
    void scalarsAndEmptyFiles() {
        try (NJScope scope = NJ.scope()) {
            assertEquals(2.5, Npy.map(NpyReaderTest.fixture("scalar.npy"), scope).getDouble());
            assertEquals(Shape.of(0, 3), Npy.map(NpyReaderTest.fixture("empty.npy"), scope).shape());
        }
    }

    @Test
    void foreignByteOrderAndTruncationAreRejected(@TempDir Path dir) throws IOException {
        try (NJScope scope = NJ.scope()) {
            assertThrows(NpyFormatException.class, () -> Npy.map(NpyReaderTest.fixture("f8_big.npy"), scope));
            assertEquals(Shape.of(2, 3), Npy.map(NpyReaderTest.fixture("u1.npy"), scope).shape(),
                    "one-byte types have no byte order");
            byte[] full = Files.readAllBytes(NpyReaderTest.fixture("f8.npy"));
            Path cut = dir.resolve("cut.npy");
            Files.write(cut, java.util.Arrays.copyOf(full, full.length - 8));
            assertThrows(NpyFormatException.class, () -> Npy.map(cut, scope));
        }
    }

    @Test
    void closedScopesCannotMap() {
        NJScope scope = NJ.scope();
        scope.close();
        assertThrows(MemoryException.class, () -> Npy.map(NpyReaderTest.fixture("f8.npy"), scope));
    }
}
