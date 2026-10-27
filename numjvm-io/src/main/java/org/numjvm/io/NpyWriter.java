package org.numjvm.io;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;

import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.MemoryErrors;
import org.numjvm.ndarray.NDArray;

/**
 * Writes arrays as NumPy {@code .npy} files (format 1.0, or 2.0 for very large headers).
 *
 * <p>C-contiguous arrays are written as they are; Fortran-contiguous arrays are written
 * with {@code fortran_order: True}, also without a copy. Any other view (transposed in
 * more than two dimensions, sliced, broadcast) is first copied into C order. Data is
 * written in native byte order and the header says so ({@code <} on little-endian
 * machines). The output for C- and Fortran-ordered arrays is byte-for-byte what NumPy's
 * {@code np.save} produces.
 */
public final class NpyWriter {

    private static final int CHUNK = 1 << 16;

    private NpyWriter() {
    }

    /**
     * Writes an array to a file, replacing it if it exists.
     *
     * @param path  the file
     * @param array the array
     * @throws NpyFormatException   if the dtype has no NumPy equivalent (bfloat16)
     * @throws UncheckedIOException if writing fails
     */
    public static void write(Path path, NDArray array) {
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path), CHUNK)) {
            write(out, array);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot write " + path, e);
        }
    }

    /**
     * Writes an array to a stream, which is not closed.
     *
     * @param out   the stream
     * @param array the array
     * @throws IOException if writing fails
     */
    public static void write(OutputStream out, NDArray array) throws IOException {
        if (!array.dtype().hasNumpyDescriptor()) {
            throw new NpyFormatException(array.dtype().typeName() + " has no NumPy equivalent and cannot be saved as .npy");
        }
        NDArray data = array;
        boolean fortran = false;
        if (!array.isCContiguous()) {
            if (array.isFContiguous()) {
                fortran = true;
            } else {
                data = array.copy();
            }
        }
        out.write(NpyHeader.encode(array.dtype().numpyDescriptor(), fortran, array.shape()));
        long bytes = data.nbytes();
        if (bytes == 0) {
            return;
        }
        MemorySegment segment = SharedSecrets.arrayAccess().buffer(data).segment();
        long offset = SharedSecrets.arrayAccess().offset(data);
        byte[] chunk = new byte[(int) Math.min(CHUNK, bytes)];
        long done = 0;
        try {
            while (done < bytes) {
                int n = (int) Math.min(chunk.length, bytes - done);
                MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, offset + done, chunk, 0, n);
                out.write(chunk, 0, n);
                done += n;
            }
        } catch (IllegalStateException | WrongThreadException e) {
            throw MemoryErrors.translate(e, data.memoryOwner());
        }
    }
}
