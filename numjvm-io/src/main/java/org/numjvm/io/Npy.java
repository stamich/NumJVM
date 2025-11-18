package org.numjvm.io;

import java.io.InputStream;
import java.nio.file.Path;

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
     * Reads one array from a stream, which is left open after the array's data.
     *
     * @param in the stream
     * @return the array
     */
    public static NDArray read(InputStream in) {
        return NpyReader.withDefaults().read(in);
    }
}
