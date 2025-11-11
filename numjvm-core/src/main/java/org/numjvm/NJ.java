package org.numjvm;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Properties;

import org.numjvm.dtype.DType;
import org.numjvm.internal.Factories;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.NativeMemoryTracker;
import org.numjvm.memory.MemoryStats;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;

/**
 * Entry point to NumJVM.
 *
 * <p>Array factories, scopes and memory statistics are reached through static methods
 * on this class.
 *
 * <p>Arrays created here use automatic memory: small arrays live on the Java heap, large
 * ones (above {@code -Dnumjvm.memory.heapThreshold}, 1 MiB by default) off-heap, and both
 * are reclaimed by the garbage collector. For deterministic release use a
 * {@linkplain #scope() scope}. All factories return C-contiguous, writeable arrays.
 *
 * <pre>{@code
 * NDArray a = NJ.array(new double[] {1, 2, 3, 4, 5, 6}, 2, 3);
 * NDArray z = NJ.zeros(DType.INT32, 4, 4);
 * NDArray r = NJ.arange(0, 12, 1);
 * }</pre>
 */
public final class NJ {

    private static final String VERSION = loadVersion();

    private NJ() {
    }

    /**
     * Returns the NumJVM version, for example {@code 0.1.0-alpha}.
     *
     * @return the library version, or {@code "unknown"} when the build did not record it
     */
    public static String version() {
        return VERSION;
    }

    private static HybridAllocator automatic() {
        return HybridAllocator.defaultInstance();
    }

    // ------------------------------------------------------------------ factories

    /**
     * Returns a new array filled with zeros.
     *
     * @param dtype element type
     * @param shape dimensions; none for a 0-D array
     * @return the array
     */
    public static NDArray zeros(DType dtype, long... shape) {
        return Factories.zeros(automatic(), dtype, shape);
    }

    /**
     * Returns a new float64 array filled with zeros.
     *
     * @param shape dimensions
     * @return the array
     */
    public static NDArray zeros(long... shape) {
        return zeros(DType.FLOAT64, shape);
    }

    /**
     * Returns a new array filled with ones.
     *
     * @param dtype element type (real)
     * @param shape dimensions
     * @return the array
     */
    public static NDArray ones(DType dtype, long... shape) {
        return full(dtype, 1.0, shape);
    }

    /**
     * Returns a new float64 array filled with ones.
     *
     * @param shape dimensions
     * @return the array
     */
    public static NDArray ones(long... shape) {
        return ones(DType.FLOAT64, shape);
    }

    /**
     * Returns a new array without specified contents.
     *
     * <p>NumJVM always zero-fills new memory (the JDK does), so this is the same as
     * {@link #zeros(DType, long...)}; it exists for NumPy familiarity.
     *
     * @param dtype element type
     * @param shape dimensions
     * @return the array
     */
    public static NDArray empty(DType dtype, long... shape) {
        return zeros(dtype, shape);
    }

    /**
     * Returns a new array with every element set to {@code value}.
     *
     * @param dtype element type (real); the value is converted like {@code setDouble}
     * @param value fill value
     * @param shape dimensions
     * @return the array
     */
    public static NDArray full(DType dtype, double value, long... shape) {
        return Factories.full(automatic(), dtype, value, shape);
    }

    /**
     * Returns a float64 array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(double[] values, long... shape) {
        return Factories.of(automatic(), values, shape);
    }

    /**
     * Returns a float32 array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(float[] values, long... shape) {
        return Factories.of(automatic(), values, shape);
    }

    /**
     * Returns an int64 array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(long[] values, long... shape) {
        return Factories.of(automatic(), values, shape);
    }

    /**
     * Returns an int32 array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(int[] values, long... shape) {
        return Factories.of(automatic(), values, shape);
    }

    /**
     * Returns a bool array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(boolean[] values, long... shape) {
        return Factories.of(automatic(), values, shape);
    }

    /**
     * Returns an array of any real dtype built from {@code double} values, each converted
     * like {@code setDouble}.
     *
     * @param dtype  element type of the result
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(DType dtype, double[] values, long... shape) {
        return Factories.of(automatic(), dtype, values, shape);
    }

    /**
     * Returns a 0-D float64 array.
     *
     * @param value the single element
     * @return the array, with shape {@code []}
     */
    public static NDArray scalar(double value) {
        NDArray a = zeros(DType.FLOAT64);
        a.setDouble(value);
        return a;
    }

    /**
     * Returns a 0-D int64 array.
     *
     * @param value the single element
     * @return the array, with shape {@code []}
     */
    public static NDArray scalar(long value) {
        NDArray a = zeros(DType.INT64);
        a.setLong(value);
        return a;
    }

    /**
     * Returns the int64 values {@code 0, 1, ..., stop - 1}.
     *
     * @param stop end, exclusive
     * @return a 1-D array
     */
    public static NDArray arange(long stop) {
        return arange(0, stop, 1);
    }

    /**
     * Returns evenly spaced int64 values in {@code [start, stop)}.
     *
     * @param start first value
     * @param stop  end, exclusive
     * @param step  spacing, positive or negative, not zero
     * @return a 1-D array
     */
    public static NDArray arange(long start, long stop, long step) {
        return Factories.arange(automatic(), start, stop, step);
    }

    /**
     * Returns evenly spaced float64 values: element {@code i} is {@code start + i * step},
     * for {@code ceil((stop - start) / step)} elements, as in NumPy.
     *
     * @param start first value
     * @param stop  end, exclusive
     * @param step  spacing, not zero
     * @return a 1-D array
     */
    public static NDArray arange(double start, double stop, double step) {
        return Factories.arange(automatic(), start, stop, step);
    }

    // ------------------------------------------------------------------ scopes and memory

    /**
     * Opens a memory scope confined to the current thread. Arrays created through the
     * scope are freed when it is closed.
     *
     * @return a new confined scope; close it with try-with-resources
     * @see NJScope
     */
    public static NJScope scope() {
        return NJScope.confined();
    }

    /**
     * Opens a memory scope confined to the current thread; same as {@link #scope()}.
     *
     * @return a new confined scope
     */
    public static NJScope scopeConfined() {
        return NJScope.confined();
    }

    /**
     * Opens a memory scope whose arrays any thread may access.
     *
     * @return a new shared scope
     */
    public static NJScope scopeShared() {
        return NJScope.shared();
    }

    /**
     * Returns a snapshot of NumJVM's native memory usage and memory settings.
     *
     * <p>Settings are read from system properties:
     * {@code -Dnumjvm.memory.heapThreshold=<bytes>} (default 1 MiB) and
     * {@code -Dnumjvm.memory.nativeSoftLimit=<bytes>} (default: none). The hard bound on
     * native memory is the JVM's {@code -XX:MaxDirectMemorySize}.
     *
     * @return current statistics
     */
    public static MemoryStats memoryStats() {
        HybridAllocator allocator = HybridAllocator.defaultInstance();
        NativeMemoryTracker t = NativeMemoryTracker.GLOBAL;
        return new MemoryStats(t.liveBytes(), t.peakBytes(), t.allocationCount(),
                allocator.heapThreshold(), allocator.nativeSoftLimit());
    }

    private static String loadVersion() {
        try (InputStream in = NJ.class.getResourceAsStream("version.properties")) {
            if (in == null) {
                return "unknown";
            }
            Properties p = new Properties();
            p.load(in);
            String v = p.getProperty("version", "unknown");
            return v.startsWith("${") ? "unknown" : v;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
