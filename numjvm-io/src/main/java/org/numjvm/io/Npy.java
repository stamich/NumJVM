package org.numjvm.io;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import org.numjvm.internal.SharedSecrets;
import org.numjvm.internal.memory.Buffer;
import org.numjvm.ndarray.Strides;

import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;

/**
 * Reads and writes NumPy {@code .npy} files with default settings.
 *
 * <pre>{@code
 * NDArray a = Npy.read(Path.of("matrix.npy"));
 * Npy.write(Path.of("out.npy"), a);
 * }</pre>
 *
 * <p>Use {@link NpyReader#withLimits(NpyLimits)} to read untrusted files with tighter
 * limits. Pickled Python objects are never loaded.
 */
public final class Npy {

    private Npy() {
    }

    /**
     * Reads a {@code .npy} file into automatically managed memory.
     *
     * @param path the file
     * @return the array, in native byte order, C- or Fortran-contiguous as stored
     * @throws NpyFormatException if the file is malformed, truncated or unsupported
     * @throws java.io.UncheckedIOException if reading fails
     */
    public static NDArray read(Path path) {
        return NpyReader.withDefaults().read(path);
    }

    /**
     * Reads a {@code .npy} file into memory owned by a scope.
     *
     * @param path  the file
     * @param scope the scope that owns the result
     * @return the array, valid until the scope closes
     */
    public static NDArray read(Path path, NJScope scope) {
        return NpyReader.withDefaults().read(path, scope);
    }

    /**
     * Writes an array as a {@code .npy} file, replacing the file if it exists.
     *
     * @param path  the file
     * @param array the array; any dtype except bfloat16, any layout
     * @throws NpyFormatException if the dtype has no NumPy equivalent
     * @throws java.io.UncheckedIOException if writing fails
     * @see NpyWriter
     */
    public static void write(Path path, NDArray array) {
        NpyWriter.write(path, array);
    }

    /**
     * Maps a {@code .npy} file into memory instead of reading it ({@code np.load(..., mmap_mode="r")}).
     *
     * <p>No data is copied: the array is a read-only view of the file's pages, which the
     * operating system loads on first access. The mapping belongs to {@code scope} and is
     * released when the scope closes; the array then becomes inaccessible. The file must
     * be in native byte order (or use one-byte elements); other files must be
     * {@linkplain #read(Path) read}.
     *
     * @param path  the file
     * @param scope the scope that owns the mapping
     * @return a read-only array backed by the file
     * @throws NpyFormatException if the file is malformed, truncated or not in native byte order
     * @throws java.io.UncheckedIOException if the file cannot be opened or mapped
     */
    public static NDArray map(Path path, NJScope scope) {
        NpyHeader h;
        long fileSize;
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path))) {
            h = NpyHeader.read(in, NpyLimits.defaults());
            fileSize = Files.size(path);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + path, e);
        }
        int unit = h.dtype().isComplex() ? h.dtype().size() / 2 : h.dtype().size();
        if (unit > 1 && h.byteOrder() != ByteOrder.nativeOrder()) {
            throw new NpyFormatException(path + " is " + h.byteOrder() + "; only native byte order can be mapped,"
                    + " use Npy.read to convert it");
        }
        long bytes = h.dataBytes();
        if (fileSize - h.dataOffset() < bytes) {
            throw new NpyFormatException("truncated .npy file: header declares " + bytes + " data bytes but only "
                    + Math.max(0, fileSize - h.dataOffset()) + " follow");
        }
        MemorySegment segment;
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            segment = ch.map(FileChannel.MapMode.READ_ONLY, h.dataOffset(), bytes,
                    SharedSecrets.scopeAccess().arena(scope));
        } catch (IOException e) {
            throw new UncheckedIOException("cannot map " + path, e);
        }
        int alignment = (int) Math.min(64, bytes == 0 ? 64 : segment.maxByteAlignment());
        Buffer buffer = Buffer.of(segment, scope, alignment);
        Strides strides = h.fortranOrder()
                ? Strides.fContiguous(h.shape(), h.dtype().size())
                : Strides.cContiguous(h.shape(), h.dtype().size());
        return SharedSecrets.arrayAccess().create(buffer, h.dtype(), 0, h.shape(), strides, false);
    }

    /**
     * Reads one array from a stream, which is left open after the array's data.
     *
     * @param in the stream
     * @return the array
     */
    public static NDArray read(InputStream in) {
        return NpyReader.withDefaults().read(in);
    }
}
