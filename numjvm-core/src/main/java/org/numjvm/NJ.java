package org.numjvm;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Properties;

import org.numjvm.dtype.Casting;
import org.numjvm.dtype.Complex;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.TypePromotion;
import org.numjvm.internal.Factories;
import org.numjvm.internal.memory.HybridAllocator;
import org.numjvm.internal.memory.NativeMemoryTracker;
import org.numjvm.memory.MemoryStats;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Reductions;
import org.numjvm.ufunc.Rounding;
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
     * @param dtype element type
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
     * @param dtype element type; the value is converted like {@code setDouble}
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
     * Returns an array of any dtype built from {@code double} values, each converted
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
     * Returns a complex128 array holding a copy of {@code values}.
     *
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     */
    public static NDArray array(Complex[] values, long... shape) {
        return Factories.of(automatic(), DType.COMPLEX128, values, shape);
    }

    /**
     * Returns a complex64 or complex128 array holding a copy of {@code values}, each
     * converted like {@code setComplex}.
     *
     * @param dtype  complex64 or complex128
     * @param values elements in C order
     * @param shape  dimensions whose product is {@code values.length}; none for 1-D
     * @return the array
     * @throws org.numjvm.exception.DTypeException if {@code dtype} is not complex
     */
    public static NDArray array(DType dtype, Complex[] values, long... shape) {
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
     * Returns a 0-D complex128 array.
     *
     * @param value the single element
     * @return the array, with shape {@code []}
     */
    public static NDArray scalar(Complex value) {
        NDArray a = zeros(DType.COMPLEX128);
        a.setComplex(value);
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

    /**
     * Element-wise {@code a - b} with broadcasting: not defined for two bool operands ({@link Ufuncs#SUBTRACT}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray subtract(NDArray a, NDArray b) {
        return Ufuncs.SUBTRACT.apply(a, b);
    }

    /**
     * Element-wise {@code a - b} with a weak scalar ({@link Ufuncs#SUBTRACT}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray subtract(NDArray a, double b) {
        return Ufuncs.SUBTRACT.apply(a, b);
    }

    /**
     * Element-wise {@code a - b} with a weak scalar ({@link Ufuncs#SUBTRACT}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray subtract(NDArray a, long b) {
        return Ufuncs.SUBTRACT.apply(a, b);
    }

    /**
     * Element-wise {@code a * b} with broadcasting: logical and for bool ({@link Ufuncs#MULTIPLY}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray multiply(NDArray a, NDArray b) {
        return Ufuncs.MULTIPLY.apply(a, b);
    }

    /**
     * Element-wise {@code a * b} with a weak scalar ({@link Ufuncs#MULTIPLY}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray multiply(NDArray a, double b) {
        return Ufuncs.MULTIPLY.apply(a, b);
    }

    /**
     * Element-wise {@code a * b} with a weak scalar ({@link Ufuncs#MULTIPLY}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray multiply(NDArray a, long b) {
        return Ufuncs.MULTIPLY.apply(a, b);
    }

    /**
     * Element-wise {@code a / b} with broadcasting: true division; integers give float64 ({@link Ufuncs#DIVIDE}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray divide(NDArray a, NDArray b) {
        return Ufuncs.DIVIDE.apply(a, b);
    }

    /**
     * Element-wise {@code a / b} with a weak scalar ({@link Ufuncs#DIVIDE}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray divide(NDArray a, double b) {
        return Ufuncs.DIVIDE.apply(a, b);
    }

    /**
     * Element-wise {@code a / b} with a weak scalar ({@link Ufuncs#DIVIDE}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray divide(NDArray a, long b) {
        return Ufuncs.DIVIDE.apply(a, b);
    }

    /**
     * Element-wise {@code floor(a / b)} with broadcasting: integer division by zero gives 0 ({@link Ufuncs#FLOOR_DIVIDE}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray floorDivide(NDArray a, NDArray b) {
        return Ufuncs.FLOOR_DIVIDE.apply(a, b);
    }

    /**
     * Element-wise {@code floor(a / b)} with a weak scalar ({@link Ufuncs#FLOOR_DIVIDE}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray floorDivide(NDArray a, double b) {
        return Ufuncs.FLOOR_DIVIDE.apply(a, b);
    }

    /**
     * Element-wise {@code floor(a / b)} with a weak scalar ({@link Ufuncs#FLOOR_DIVIDE}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray floorDivide(NDArray a, long b) {
        return Ufuncs.FLOOR_DIVIDE.apply(a, b);
    }

    /**
     * Element-wise {@code a % b} with broadcasting: Python-style remainder with the sign of the divisor ({@link Ufuncs#REMAINDER}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray remainder(NDArray a, NDArray b) {
        return Ufuncs.REMAINDER.apply(a, b);
    }

    /**
     * Element-wise {@code a % b} with a weak scalar ({@link Ufuncs#REMAINDER}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray remainder(NDArray a, double b) {
        return Ufuncs.REMAINDER.apply(a, b);
    }

    /**
     * Element-wise {@code a % b} with a weak scalar ({@link Ufuncs#REMAINDER}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray remainder(NDArray a, long b) {
        return Ufuncs.REMAINDER.apply(a, b);
    }

    /**
     * Element-wise {@code a ** b} with broadcasting: integer powers wrap ({@link Ufuncs#POWER}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray power(NDArray a, NDArray b) {
        return Ufuncs.POWER.apply(a, b);
    }

    /**
     * Element-wise {@code a ** b} with a weak scalar ({@link Ufuncs#POWER}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray power(NDArray a, double b) {
        return Ufuncs.POWER.apply(a, b);
    }

    /**
     * Element-wise {@code a ** b} with a weak scalar ({@link Ufuncs#POWER}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray power(NDArray a, long b) {
        return Ufuncs.POWER.apply(a, b);
    }

    /**
     * Element-wise {@code max(a, b)} with broadcasting: NaN propagates ({@link Ufuncs#MAXIMUM}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray maximum(NDArray a, NDArray b) {
        return Ufuncs.MAXIMUM.apply(a, b);
    }

    /**
     * Element-wise {@code max(a, b)} with a weak scalar ({@link Ufuncs#MAXIMUM}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray maximum(NDArray a, double b) {
        return Ufuncs.MAXIMUM.apply(a, b);
    }

    /**
     * Element-wise {@code max(a, b)} with a weak scalar ({@link Ufuncs#MAXIMUM}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray maximum(NDArray a, long b) {
        return Ufuncs.MAXIMUM.apply(a, b);
    }

    /**
     * Element-wise {@code min(a, b)} with broadcasting: NaN propagates ({@link Ufuncs#MINIMUM}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray minimum(NDArray a, NDArray b) {
        return Ufuncs.MINIMUM.apply(a, b);
    }

    /**
     * Element-wise {@code min(a, b)} with a weak scalar ({@link Ufuncs#MINIMUM}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray minimum(NDArray a, double b) {
        return Ufuncs.MINIMUM.apply(a, b);
    }

    /**
     * Element-wise {@code min(a, b)} with a weak scalar ({@link Ufuncs#MINIMUM}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray minimum(NDArray a, long b) {
        return Ufuncs.MINIMUM.apply(a, b);
    }

    /**
     * Element-wise {@code fmax(a, b)} with broadcasting: a NaN operand is ignored ({@link Ufuncs#FMAX}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray fmax(NDArray a, NDArray b) {
        return Ufuncs.FMAX.apply(a, b);
    }

    /**
     * Element-wise {@code fmax(a, b)} with a weak scalar ({@link Ufuncs#FMAX}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray fmax(NDArray a, double b) {
        return Ufuncs.FMAX.apply(a, b);
    }

    /**
     * Element-wise {@code fmax(a, b)} with a weak scalar ({@link Ufuncs#FMAX}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray fmax(NDArray a, long b) {
        return Ufuncs.FMAX.apply(a, b);
    }

    /**
     * Element-wise {@code fmin(a, b)} with broadcasting: a NaN operand is ignored ({@link Ufuncs#FMIN}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray fmin(NDArray a, NDArray b) {
        return Ufuncs.FMIN.apply(a, b);
    }

    /**
     * Element-wise {@code fmin(a, b)} with a weak scalar ({@link Ufuncs#FMIN}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray fmin(NDArray a, double b) {
        return Ufuncs.FMIN.apply(a, b);
    }

    /**
     * Element-wise {@code fmin(a, b)} with a weak scalar ({@link Ufuncs#FMIN}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray fmin(NDArray a, long b) {
        return Ufuncs.FMIN.apply(a, b);
    }

    /**
     * Element-wise {@code a == b} with broadcasting: giving bool ({@link Ufuncs#EQUAL}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    @SuppressWarnings("java:S1221") // NumPy's name for the ufunc; not Object.equals
    public static NDArray equal(NDArray a, NDArray b) {
        return Ufuncs.EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a == b} with a weak scalar ({@link Ufuncs#EQUAL}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    @SuppressWarnings("java:S1221") // NumPy's name for the ufunc; not Object.equals
    public static NDArray equal(NDArray a, double b) {
        return Ufuncs.EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a == b} with a weak scalar ({@link Ufuncs#EQUAL}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    @SuppressWarnings("java:S1221") // NumPy's name for the ufunc; not Object.equals
    public static NDArray equal(NDArray a, long b) {
        return Ufuncs.EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a != b} with broadcasting: giving bool ({@link Ufuncs#NOT_EQUAL}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray notEqual(NDArray a, NDArray b) {
        return Ufuncs.NOT_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a != b} with a weak scalar ({@link Ufuncs#NOT_EQUAL}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray notEqual(NDArray a, double b) {
        return Ufuncs.NOT_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a != b} with a weak scalar ({@link Ufuncs#NOT_EQUAL}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray notEqual(NDArray a, long b) {
        return Ufuncs.NOT_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a < b} with broadcasting: giving bool ({@link Ufuncs#LESS}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray less(NDArray a, NDArray b) {
        return Ufuncs.LESS.apply(a, b);
    }

    /**
     * Element-wise {@code a < b} with a weak scalar ({@link Ufuncs#LESS}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray less(NDArray a, double b) {
        return Ufuncs.LESS.apply(a, b);
    }

    /**
     * Element-wise {@code a < b} with a weak scalar ({@link Ufuncs#LESS}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray less(NDArray a, long b) {
        return Ufuncs.LESS.apply(a, b);
    }

    /**
     * Element-wise {@code a <= b} with broadcasting: giving bool ({@link Ufuncs#LESS_EQUAL}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray lessEqual(NDArray a, NDArray b) {
        return Ufuncs.LESS_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a <= b} with a weak scalar ({@link Ufuncs#LESS_EQUAL}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray lessEqual(NDArray a, double b) {
        return Ufuncs.LESS_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a <= b} with a weak scalar ({@link Ufuncs#LESS_EQUAL}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray lessEqual(NDArray a, long b) {
        return Ufuncs.LESS_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a > b} with broadcasting: giving bool ({@link Ufuncs#GREATER}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray greater(NDArray a, NDArray b) {
        return Ufuncs.GREATER.apply(a, b);
    }

    /**
     * Element-wise {@code a > b} with a weak scalar ({@link Ufuncs#GREATER}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray greater(NDArray a, double b) {
        return Ufuncs.GREATER.apply(a, b);
    }

    /**
     * Element-wise {@code a > b} with a weak scalar ({@link Ufuncs#GREATER}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray greater(NDArray a, long b) {
        return Ufuncs.GREATER.apply(a, b);
    }

    /**
     * Element-wise {@code a >= b} with broadcasting: giving bool ({@link Ufuncs#GREATER_EQUAL}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray greaterEqual(NDArray a, NDArray b) {
        return Ufuncs.GREATER_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a >= b} with a weak scalar ({@link Ufuncs#GREATER_EQUAL}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray greaterEqual(NDArray a, double b) {
        return Ufuncs.GREATER_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a >= b} with a weak scalar ({@link Ufuncs#GREATER_EQUAL}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray greaterEqual(NDArray a, long b) {
        return Ufuncs.GREATER_EQUAL.apply(a, b);
    }

    /**
     * Element-wise {@code a and b} with broadcasting: giving bool; non-zero is true ({@link Ufuncs#LOGICAL_AND}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray logicalAnd(NDArray a, NDArray b) {
        return Ufuncs.LOGICAL_AND.apply(a, b);
    }

    /**
     * Element-wise {@code a and b} with a weak scalar ({@link Ufuncs#LOGICAL_AND}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray logicalAnd(NDArray a, double b) {
        return Ufuncs.LOGICAL_AND.apply(a, b);
    }

    /**
     * Element-wise {@code a and b} with a weak scalar ({@link Ufuncs#LOGICAL_AND}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray logicalAnd(NDArray a, long b) {
        return Ufuncs.LOGICAL_AND.apply(a, b);
    }

    /**
     * Element-wise {@code a or b} with broadcasting: giving bool; non-zero is true ({@link Ufuncs#LOGICAL_OR}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray logicalOr(NDArray a, NDArray b) {
        return Ufuncs.LOGICAL_OR.apply(a, b);
    }

    /**
     * Element-wise {@code a or b} with a weak scalar ({@link Ufuncs#LOGICAL_OR}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray logicalOr(NDArray a, double b) {
        return Ufuncs.LOGICAL_OR.apply(a, b);
    }

    /**
     * Element-wise {@code a or b} with a weak scalar ({@link Ufuncs#LOGICAL_OR}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray logicalOr(NDArray a, long b) {
        return Ufuncs.LOGICAL_OR.apply(a, b);
    }

    /**
     * Element-wise {@code a xor b} with broadcasting: giving bool; non-zero is true ({@link Ufuncs#LOGICAL_XOR}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray logicalXor(NDArray a, NDArray b) {
        return Ufuncs.LOGICAL_XOR.apply(a, b);
    }

    /**
     * Element-wise {@code a xor b} with a weak scalar ({@link Ufuncs#LOGICAL_XOR}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray logicalXor(NDArray a, double b) {
        return Ufuncs.LOGICAL_XOR.apply(a, b);
    }

    /**
     * Element-wise {@code a xor b} with a weak scalar ({@link Ufuncs#LOGICAL_XOR}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray logicalXor(NDArray a, long b) {
        return Ufuncs.LOGICAL_XOR.apply(a, b);
    }

    /**
     * Element-wise {@code not x}: giving bool; non-zero is true ({@link Ufuncs#LOGICAL_NOT}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray logicalNot(NDArray x) {
        return Ufuncs.LOGICAL_NOT.apply(x);
    }

    /**
     * Element-wise {@code a & b} with broadcasting: bool and integers ({@link Ufuncs#BITWISE_AND}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray bitwiseAnd(NDArray a, NDArray b) {
        return Ufuncs.BITWISE_AND.apply(a, b);
    }

    /**
     * Element-wise {@code a & b} with a weak scalar ({@link Ufuncs#BITWISE_AND}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray bitwiseAnd(NDArray a, double b) {
        return Ufuncs.BITWISE_AND.apply(a, b);
    }

    /**
     * Element-wise {@code a & b} with a weak scalar ({@link Ufuncs#BITWISE_AND}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray bitwiseAnd(NDArray a, long b) {
        return Ufuncs.BITWISE_AND.apply(a, b);
    }

    /**
     * Element-wise {@code a | b} with broadcasting: bool and integers ({@link Ufuncs#BITWISE_OR}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray bitwiseOr(NDArray a, NDArray b) {
        return Ufuncs.BITWISE_OR.apply(a, b);
    }

    /**
     * Element-wise {@code a | b} with a weak scalar ({@link Ufuncs#BITWISE_OR}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray bitwiseOr(NDArray a, double b) {
        return Ufuncs.BITWISE_OR.apply(a, b);
    }

    /**
     * Element-wise {@code a | b} with a weak scalar ({@link Ufuncs#BITWISE_OR}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray bitwiseOr(NDArray a, long b) {
        return Ufuncs.BITWISE_OR.apply(a, b);
    }

    /**
     * Element-wise {@code a ^ b} with broadcasting: bool and integers ({@link Ufuncs#BITWISE_XOR}).
     *
     * @param a first operand
     * @param b second operand
     * @return a new array with the broadcast shape
     */
    public static NDArray bitwiseXor(NDArray a, NDArray b) {
        return Ufuncs.BITWISE_XOR.apply(a, b);
    }

    /**
     * Element-wise {@code a ^ b} with a weak scalar ({@link Ufuncs#BITWISE_XOR}).
     *
     * @param a the array
     * @param b the scalar; float and complex arrays keep their dtype, others become float64
     * @return a new array
     */
    public static NDArray bitwiseXor(NDArray a, double b) {
        return Ufuncs.BITWISE_XOR.apply(a, b);
    }

    /**
     * Element-wise {@code a ^ b} with a weak scalar ({@link Ufuncs#BITWISE_XOR}).
     *
     * @param a the array
     * @param b the scalar; it must fit the array's integer dtype, bool arrays become int64
     * @return a new array
     */
    public static NDArray bitwiseXor(NDArray a, long b) {
        return Ufuncs.BITWISE_XOR.apply(a, b);
    }

    /**
     * Element-wise {@code ~x}: bitwise not; logical not for bool ({@link Ufuncs#INVERT}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray invert(NDArray x) {
        return Ufuncs.INVERT.apply(x);
    }

    /**
     * Element-wise {@code -x}: integers wrap; not defined for bool ({@link Ufuncs#NEGATIVE}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray negative(NDArray x) {
        return Ufuncs.NEGATIVE.apply(x);
    }

    /**
     * Element-wise {@code +x}: a copy; not defined for bool ({@link Ufuncs#POSITIVE}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray positive(NDArray x) {
        return Ufuncs.POSITIVE.apply(x);
    }

    /**
     * Element-wise {@code |x|}: abs(MIN) wraps to MIN ({@link Ufuncs#ABSOLUTE}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray abs(NDArray x) {
        return Ufuncs.ABSOLUTE.apply(x);
    }

    /**
     * Element-wise {@code sign(x)}: -1, 0 or 1; NaN for NaN ({@link Ufuncs#SIGN}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray sign(NDArray x) {
        return Ufuncs.SIGN.apply(x);
    }

    /**
     * Element-wise {@code x * x}: integers wrap ({@link Ufuncs#SQUARE}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray square(NDArray x) {
        return Ufuncs.SQUARE.apply(x);
    }

    /**
     * Element-wise {@code sqrt(x)}: integers give the first float dtype they convert to safely ({@link Ufuncs#SQRT}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray sqrt(NDArray x) {
        return Ufuncs.SQRT.apply(x);
    }

    /**
     * Element-wise {@code exp(x)}: integers give the first float dtype they convert to safely ({@link Ufuncs#EXP}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray exp(NDArray x) {
        return Ufuncs.EXP.apply(x);
    }

    /**
     * Element-wise {@code log(x)}: integers give the first float dtype they convert to safely ({@link Ufuncs#LOG}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray log(NDArray x) {
        return Ufuncs.LOG.apply(x);
    }

    /**
     * Element-wise {@code sin(x)}: integers give the first float dtype they convert to safely ({@link Ufuncs#SIN}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray sin(NDArray x) {
        return Ufuncs.SIN.apply(x);
    }

    /**
     * Element-wise {@code cos(x)}: integers give the first float dtype they convert to safely ({@link Ufuncs#COS}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray cos(NDArray x) {
        return Ufuncs.COS.apply(x);
    }

    /**
     * Element-wise {@code tanh(x)}: integers give the first float dtype they convert to safely ({@link Ufuncs#TANH}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray tanh(NDArray x) {
        return Ufuncs.TANH.apply(x);
    }

    /**
     * Element-wise {@code trunc(x)}: integers keep their dtype ({@link Ufuncs#TRUNC}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray trunc(NDArray x) {
        return Ufuncs.TRUNC.apply(x);
    }

    /**
     * Element-wise {@code rint(x)}: halves to even; integers give a float dtype ({@link Ufuncs#RINT}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray rint(NDArray x) {
        return Ufuncs.RINT.apply(x);
    }

    /**
     * Rounds to the nearest integer, halves to even ({@link Rounding#round(NDArray)}).
     *
     * @param x the array
     * @return a new array; integer arrays keep their dtype
     */
    public static NDArray round(NDArray x) {
        return Rounding.round(x);
    }

    /**
     * Rounds to {@code decimals} decimal places, halves to even, as NumPy's {@code np.round}
     * ({@link Rounding#round(NDArray, int)}).
     *
     * @param x        the array
     * @param decimals decimal places; negative values round to tens, hundreds, ...
     * @return a new array of the same dtype
     */
    public static NDArray round(NDArray x, int decimals) {
        return Rounding.round(x, decimals);
    }

    /**
     * Element-wise {@code floor(x)}: integers keep their dtype ({@link Ufuncs#FLOOR}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray floor(NDArray x) {
        return Ufuncs.FLOOR.apply(x);
    }

    /**
     * Element-wise {@code ceil(x)}: integers keep their dtype ({@link Ufuncs#CEIL}).
     *
     * @param x the input
     * @return a new array with the shape of {@code x}
     */
    public static NDArray ceil(NDArray x) {
        return Ufuncs.CEIL.apply(x);
    }

    /**
     * Sum over all elements ({@link Reductions#sum(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray sum(NDArray x) {
        return Reductions.sum(x);
    }

    /**
     * Sum over one axis ({@link Reductions#sum(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray sum(NDArray x, int axis) {
        return Reductions.sum(x, axis);
    }

    /**
     * Sum over several axes ({@link Reductions#sum(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray sum(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.sum(x, axes, keepdims);
    }

    /**
     * Product over all elements ({@link Reductions#prod(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray prod(NDArray x) {
        return Reductions.prod(x);
    }

    /**
     * Product over one axis ({@link Reductions#prod(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray prod(NDArray x, int axis) {
        return Reductions.prod(x, axis);
    }

    /**
     * Product over several axes ({@link Reductions#prod(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray prod(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.prod(x, axes, keepdims);
    }

    /**
     * Maximum over all elements ({@link Reductions#max(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray max(NDArray x) {
        return Reductions.max(x);
    }

    /**
     * Maximum over one axis ({@link Reductions#max(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray max(NDArray x, int axis) {
        return Reductions.max(x, axis);
    }

    /**
     * Maximum over several axes ({@link Reductions#max(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray max(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.max(x, axes, keepdims);
    }

    /**
     * Minimum over all elements ({@link Reductions#min(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray min(NDArray x) {
        return Reductions.min(x);
    }

    /**
     * Minimum over one axis ({@link Reductions#min(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray min(NDArray x, int axis) {
        return Reductions.min(x, axis);
    }

    /**
     * Minimum over several axes ({@link Reductions#min(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray min(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.min(x, axes, keepdims);
    }

    /**
     * Mean over all elements ({@link Reductions#mean(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray mean(NDArray x) {
        return Reductions.mean(x);
    }

    /**
     * Mean over one axis ({@link Reductions#mean(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray mean(NDArray x, int axis) {
        return Reductions.mean(x, axis);
    }

    /**
     * Mean over several axes ({@link Reductions#mean(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray mean(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.mean(x, axes, keepdims);
    }

    /**
     * Whether any element is true over all elements ({@link Reductions#any(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray any(NDArray x) {
        return Reductions.any(x);
    }

    /**
     * Whether any element is true over one axis ({@link Reductions#any(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray any(NDArray x, int axis) {
        return Reductions.any(x, axis);
    }

    /**
     * Whether any element is true over several axes ({@link Reductions#any(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray any(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.any(x, axes, keepdims);
    }

    /**
     * Whether every element is true over all elements ({@link Reductions#all(NDArray)}).
     *
     * @param x the array
     * @return a 0-D array
     */
    public static NDArray all(NDArray x) {
        return Reductions.all(x);
    }

    /**
     * Whether every element is true over one axis ({@link Reductions#all(NDArray, int)}).
     *
     * @param x    the array
     * @param axis the axis; negative values count from the end
     * @return the result, with that axis removed
     */
    public static NDArray all(NDArray x, int axis) {
        return Reductions.all(x, axis);
    }

    /**
     * Whether every element is true over several axes ({@link Reductions#all(NDArray, int[], boolean)}).
     *
     * @param x        the array
     * @param axes     the axes; {@code null} for all, an empty array for none
     * @param keepdims keep reduced axes with length 1
     * @return the result
     */
    public static NDArray all(NDArray x, int[] axes, boolean keepdims) {
        return Reductions.all(x, axes, keepdims);
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
