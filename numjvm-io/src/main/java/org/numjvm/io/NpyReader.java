package org.numjvm.io;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.MemoryAllocator;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;

/**
 * Reads NumPy {@code .npy} files into arrays.
 *
 * <p>The data is copied into a new array in native byte order: big-endian files are
 * byte-swapped while reading. Fortran-ordered files become Fortran-contiguous arrays
 * without a reordering copy. Before allocating, the reader checks the declared size
 * against its {@link NpyLimits} and, for files, against the bytes actually present.
 */
public final class NpyReader {

    private static final int CHUNK = 1 << 16;

    private final NpyLimits limits;

    private NpyReader(NpyLimits limits) {
        this.limits = Objects.requireNonNull(limits, "limits");
    }

    /** @return a reader with {@link NpyLimits#defaults()} */
    public static NpyReader withDefaults() {
        return new NpyReader(NpyLimits.defaults());
    }

    /**
     * @param limits limits on header size, rank and payload size
     * @return a reader that enforces them
     */
    public static NpyReader withLimits(NpyLimits limits) {
        return new NpyReader(limits);
    }

    /**
     * Reads a file into automatically managed memory.
     *
     * @param path the file
     * @return the array
     * @throws NpyFormatException if the file is malformed, truncated or over a limit
     * @throws UncheckedIOException if reading fails
     */
    public NDArray read(Path path) {
        return read(path, HybridAllocator.defaultInstance());
    }

    /**
     * Reads a file into memory owned by {@code scope}.
     *
     * @param path  the file
     * @param scope the scope that will own the array
     * @return the array, valid until the scope closes
     */
    public NDArray read(Path path, NJScope scope) {
        return read(path, SharedSecrets.scopeAccess().allocator(scope));
    }

    /**
     * Reads one array from a stream into automatically managed memory. The stream is left
     * positioned after the array's data and is not closed.
     *
     * @param in the stream
     * @return the array
     */
    public NDArray read(InputStream in) {
        try {
            return read(in, -1, HybridAllocator.defaultInstance());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private NDArray read(Path path, MemoryAllocator allocator) {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path), CHUNK)) {
            return read(in, Files.size(path), allocator);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + path, e);
        }
    }

    /** @param available total stream length if known, else -1 */
    private NDArray read(InputStream in, long available, MemoryAllocator allocator) throws IOException {
        NpyHeader h = NpyHeader.read(in, limits);
        long bytes = h.dataBytes();
        if (available >= 0 && available - h.dataOffset() < bytes) {
            throw new NpyFormatException("truncated .npy file: header declares " + bytes + " data bytes for shape "
                    + h.shape() + " but only " + Math.max(0, available - h.dataOffset()) + " follow");
        }
        NDArray array = SharedSecrets.arrayAccess().allocate(allocator, h.dtype(), h.shape(),
                h.fortranOrder() ? Order.F : Order.C);
        MemorySegment target = SharedSecrets.arrayAccess().buffer(array).segment();
        int unit = h.dtype().isComplex() ? h.dtype().size() / 2 : h.dtype().size();
        boolean swap = unit > 1 && h.byteOrder() != ByteOrder.nativeOrder();
        byte[] chunk = new byte[(int) Math.min(CHUNK, Math.max(bytes, 1))];
        long done = 0;
        while (done < bytes) {
            int want = (int) Math.min(chunk.length, bytes - done);
            int got = in.readNBytes(chunk, 0, want);
            if (got < want) {
                throw new NpyFormatException("truncated .npy file: expected " + bytes + " data bytes, found "
                        + (done + got));
            }
            if (swap) {
                swapBytes(chunk, got, unit);
            }
            MemorySegment.copy(chunk, 0, target, ValueLayout.JAVA_BYTE, done, got);
            done += got;
        }
        return array;
    }

    /** Reverses every group of {@code unit} bytes; {@code length} is a multiple of {@code unit}. */
    static void swapBytes(byte[] b, int length, int unit) {
        for (int base = 0; base < length; base += unit) {
            for (int i = 0, j = unit - 1; i < j; i++, j--) {
                byte t = b[base + i];
                b[base + i] = b[base + j];
                b[base + j] = t;
            }
        }
    }
}
