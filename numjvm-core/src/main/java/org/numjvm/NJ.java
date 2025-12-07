package org.numjvm;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Properties;

import org.numjvm.dtype.Casting;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.internal.Factories;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.NativeMemoryTracker;
import org.numjvm.memory.MemoryStats;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;

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

    // ------------------------------------------------------------------ comparison

    /**
     * Tells whether two arrays have the same shape and equal elements
     * ({@code NUMPY_COMPATIBILITY}: {@code np.array_equal}).
     *
     * <p>The dtypes may differ: values are compared as numbers ({@code 1} equals
     * {@code 1.0}). Integer and bool arrays are compared exactly as 64-bit integers
     * (uint64 by unsigned value); NaN is never equal to anything.
     *
     * @param a first array
     * @param b second array
     * @return {@code true} if shapes and all elements are equal
     */
    public static boolean arrayEquals(NDArray a, NDArray b) {
        return Comparisons.arrayEquals(a, b);
    }

    /**
     * Tells whether two arrays of the same shape are element-wise equal within a
     * tolerance, with NumPy's defaults {@code rtol = 1e-5}, {@code atol = 1e-8}.
     *
     * @param a first array
     * @param b second array
     * @return {@code true} if {@code |a - b| <= atol + rtol * |b|} for every element
     * @see #allClose(NDArray, NDArray, double, double)
     */
    public static boolean allClose(NDArray a, NDArray b) {
        return allClose(a, b, 1e-5, 1e-8);
    }

    /**
     * Tells whether two arrays of the same shape are element-wise equal within a
     * tolerance ({@code NUMPY_COMPATIBILITY}: {@code np.allclose}, which is asymmetric in
     * {@code a} and {@code b}). NaN is never close; infinities are close only to an
     * infinity of the same sign.
     *
     * @param a    first array
     * @param b    second array, the reference
     * @param rtol relative tolerance
     * @param atol absolute tolerance
     * @return {@code true} if {@code |a - b| <= atol + rtol * |b|} for every element
     * @throws org.numjvm.exception.ShapeException if the shapes differ
     */
    public static boolean allClose(NDArray a, NDArray b, double rtol, double atol) {
        return Comparisons.allClose(a, b, rtol, atol);
    }

    // ------------------------------------------------------------------ arithmetic

    /**
     * Element-wise {@code a + b} with broadcasting; logical or for bool
     * ({@code ARRAY_API}: {@code add}; {@link Ufuncs#ADD}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape and the promoted dtype
     */
    public static NDArray add(NDArray a, NDArray b) {
        return Ufuncs.ADD.apply(a, b);
    }

    /**
     * Element-wise {@code a + b} with a weak scalar ({@link Ufuncs#ADD}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray add(NDArray a, double b) {
        return Ufuncs.ADD.apply(a, b);
    }

    /**
     * Element-wise {@code a + b} with a weak scalar ({@link Ufuncs#ADD}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray add(NDArray a, long b) {
        return Ufuncs.ADD.apply(a, b);
    }

    // ------------------------------------------------------------------ type promotion

    /**
     * Returns the dtype an operation on these dtypes produces ({@code ARRAY_API}:
     * {@code result_type}; rules in {@link TypePromotion} and ADR-005).
     *
     * @param dtypes one or more dtypes
     * @return the promoted dtype
     */
    public static DType resultType(DType... dtypes) {
        return TypePromotion.resultType(dtypes);
    }

    /**
     * Returns the dtype an operation on these arrays produces ({@code ARRAY_API}:
     * {@code result_type}). Only the dtypes matter; 0-D arrays are not weak scalars.
     *
     * @param arrays one or more arrays
     * @return the promoted dtype
     */
    public static DType resultType(NDArray... arrays) {
        DType[] dtypes = new DType[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            dtypes[i] = arrays[i].dtype();
        }
        return TypePromotion.resultType(dtypes);
    }

    /**
     * Tells whether every value of {@code from} is representable in {@code to}
     * ({@code ARRAY_API}: {@code can_cast}).
     *
     * @param from source dtype
     * @param to   target dtype
     * @return {@code true} for a safe conversion
     */
    public static boolean canCast(DType from, DType to) {
        return TypePromotion.canCast(from, to);
    }

    /**
     * Tells whether {@code from} can be converted to {@code to} under a casting mode
     * ({@code NUMPY_COMPATIBILITY}: {@code np.can_cast}).
     *
     * @param from    source dtype
     * @param to      target dtype
     * @param casting the casting mode
     * @return {@code true} if the conversion is allowed
     */
    public static boolean canCast(DType from, DType to, Casting casting) {
        return TypePromotion.canCast(from, to, casting);
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
