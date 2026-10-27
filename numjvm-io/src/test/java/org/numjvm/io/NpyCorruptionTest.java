package org.numjvm.io;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.ndarray.NDArray;

/**
 * Corrupted and truncated input must fail with {@link NpyFormatException}: never with an
 * unrelated exception, an unbounded allocation or a silently wrong array.
 */
class NpyCorruptionTest {

    @ParameterizedTest
    @ValueSource(strings = {"f8", "c16_fortran", "b1", "u8_max", "scalar", "empty"})
    void everyTruncationIsDetected(String name, @TempDir Path dir) throws IOException {
        byte[] full = Files.readAllBytes(NpyReaderTest.fixture(name + ".npy"));
        for (int len = 0; len < full.length; len++) {
            byte[] cut = java.util.Arrays.copyOf(full, len);
            int length = len;
            assertThrows(NpyFormatException.class, () -> Npy.read(new ByteArrayInputStream(cut)),
                    () -> name + " truncated to " + length + " bytes (stream)");
            Path f = dir.resolve("cut.npy");
            Files.write(f, cut);
            assertThrows(NpyFormatException.class, () -> Npy.read(f),
                    () -> name + " truncated to " + length + " bytes (file)");
        }
    }

    @Test
    void randomHeaderCorruptionNeverEscapesAsAnotherException() throws IOException {
        byte[] original = Files.readAllBytes(NpyReaderTest.fixture("i4.npy"));
        Random rnd = new Random(20251118);
        int headerEnd = 128;
        for (int trial = 0; trial < 5000; trial++) {
            byte[] b = original.clone();
            int flips = 1 + rnd.nextInt(4);
            for (int k = 0; k < flips; k++) {
                b[rnd.nextInt(headerEnd)] = (byte) rnd.nextInt(256);
            }
            try {
                NDArray a = Npy.read(new ByteArrayInputStream(b));
                if (a.nbytes() > original.length) {
                    fail("corrupted header produced an array larger than the file: " + a.shape());
                }
            } catch (NpyFormatException expected) {
                // the normal outcome
            } catch (RuntimeException e) {
                fail("trial " + trial + " escaped as " + e, e);
            }
        }
    }

    @Test
    void declaredSizeLargerThanTheFileIsRejectedBeforeAllocation(@TempDir Path dir) throws IOException {
        byte[] header = NpyHeader.encode("<f8", false, org.numjvm.ndarray.Shape.of(1L << 33));  // 64 GiB declared
        Path f = dir.resolve("liar.npy");
        Files.write(f, header);
        assertThrows(NpyFormatException.class, () -> Npy.read(f));
    }
}
