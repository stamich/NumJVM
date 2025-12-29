package org.numjvm.dtype;

import java.util.Objects;

import org.numjvm.exception.DTypeException;

/**
 * Type promotion and casting rules (ADR-005): which dtype an operation on two dtypes
 * produces, and which conversions a casting mode allows.
 *
 * <p>The rules follow the Python Array API Standard, revision v2025.12, where it defines
 * a result ({@code ARRAY_API}), and NumPy 2 where it does not ({@code NUMPY_COMPATIBILITY}):
 * <ul>
 *   <li>same kind: the wider dtype; signed with unsigned: the smallest signed integer
 *       holding both ({@code int8 + uint8 -> int16}); {@code int64 + uint64 -> float64};</li>
 *   <li>bool with anything: the other dtype;</li>
 *   <li>integer with float: the smallest float, at least as wide as the float operand,
 *       that holds every value of the integer ({@code int8 + float16 -> float16},
 *       {@code int16 + float16 -> float32}, {@code int32 + float32 -> float64});</li>
 *   <li>real with complex: the complex dtype of the promoted real components
 *       ({@code float64 + complex64 -> complex128});</li>
 *   <li>bfloat16 ({@code NUMJVM_SPECIFIC}, as the {@code ml_dtypes} package): promotes like
 *       float16, except {@code bfloat16 + float16 -> float32}.</li>
 * </ul>
 */
public final class TypePromotion {

    private TypePromotion() {
    }

    /**
     * Returns the dtype of an operation on two operands of these dtypes.
     *
     * @param a first dtype
     * @param b second dtype
     * @return the promoted dtype
     */
    public static DType resultType(DType a, DType b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        if (a == b) {
            return a;
        }
        if (a.isBool()) {
            return b;
        }
        if (b.isBool()) {
            return a;
        }
        if (a.isComplex() || b.isComplex()) {
            DType real = resultType(a.isComplex() ? component(a) : a, b.isComplex() ? component(b) : b);
            return real == DType.FLOAT64 ? DType.COMPLEX128 : DType.COMPLEX64;
        }
        if (a.isFloatingPoint() && b.isFloatingPoint()) {
            return widerFloat(a, b);
        }
        if (a.isFloatingPoint() || b.isFloatingPoint()) {
            DType f = a.isFloatingPoint() ? a : b;
            DType i = a.isFloatingPoint() ? b : a;
            DType holds = switch (i.size()) {
                case 1 -> DType.FLOAT16;
                case 2 -> DType.FLOAT32;
                default -> DType.FLOAT64;
            };
            if (holds == DType.FLOAT16) {
                return f;
            }
            return f.size() >= holds.size() ? f : holds;
        }
        if (a.isUnsigned() == b.isUnsigned()) {
            return a.size() >= b.size() ? a : b;
        }
        DType signed = a.isUnsigned() ? b : a;
        DType unsigned = a.isUnsigned() ? a : b;
        if (signed.size() > unsigned.size()) {
            return signed;
        }
        return switch (unsigned.size()) {
            case 1 -> DType.INT16;
            case 2 -> DType.INT32;
            case 4 -> DType.INT64;
            default -> DType.FLOAT64;
        };
    }

    /**
     * Returns the dtype of an operation on operands of all these dtypes
     * ({@code NUMPY_COMPATIBILITY}: {@code np.result_type}).
     *
     * <p>As in NumPy, the operands are not promoted left to right: the dtype with the
     * highest NumPy type number (float16 ranks highest, bfloat16 above it as a user-defined
     * dtype would) is the starting point, and the others are promoted into it in order. So
     * {@code resultType(INT16, UINT16, FLOAT32)} is float32, although
     * {@code resultType(resultType(INT16, UINT16), FLOAT32)} is float64. For two dtypes both
     * orders agree.
     *
     * @param dtypes one or more dtypes
     * @return the promoted dtype
     * @throws DTypeException if no dtype is given
     */
    public static DType resultType(DType... dtypes) {
        if (dtypes.length == 0) {
            throw new DTypeException("resultType needs at least one dtype");
        }
        int main = 0;
        for (int i = 1; i < dtypes.length; i++) {
            if (typeNumber(dtypes[i]) > typeNumber(dtypes[main])) {
                main = i;
            }
        }
        DType r = Objects.requireNonNull(dtypes[main], "dtype");
        for (int i = 0; i < dtypes.length; i++) {
            if (i != main) {
                r = resultType(r, dtypes[i]);
            }
        }
        return r;
    }

    /**
     * NumPy's type numbers, which decide where a multi-operand promotion starts.
     */
    private static int typeNumber(DType t) {
        return switch (t) {
            case BOOL -> 0;
            case INT8 -> 1;
            case UINT8 -> 2;
            case INT16 -> 3;
            case UINT16 -> 4;
            case INT32 -> 5;
            case UINT32 -> 6;
            case INT64 -> 7;
            case UINT64 -> 8;
            case FLOAT32 -> 11;
            case FLOAT64 -> 12;
            case COMPLEX64 -> 14;
            case COMPLEX128 -> 15;
            case FLOAT16 -> 23;
            case BFLOAT16 -> 256;
        };
    }

    /**
     * Returns the dtype of an operation between an array of dtype {@code array} and a Java
     * {@code long} scalar, which is weak (ADR-005): integer arrays keep their dtype, bool
     * arrays become int64, float and complex arrays keep theirs.
     *
     * <p>The value is not inspected; operations check that it fits the result.
     *
     * @param array the array's dtype
     * @return the promoted dtype
     */
    public static DType resultTypeWithLong(DType array) {
        return array.isBool() ? DType.INT64 : array;
    }

    /**
     * Returns the dtype of an operation between an array of dtype {@code array} and a Java
     * {@code double} scalar, which is weak (ADR-005): float and complex arrays keep their
     * dtype, bool and integer arrays become float64.
     *
     * @param array the array's dtype
     * @return the promoted dtype
     */
    public static DType resultTypeWithDouble(DType array) {
        return array.isFloatingPoint() || array.isComplex() ? array : DType.FLOAT64;
    }

    /**
     * Returns the dtype of an operation between an array of dtype {@code array} and a Java
     * {@link Complex} scalar, which is weak like Python's {@code complex} (ADR-005): complex
     * arrays keep their dtype, float16, bfloat16 and float32 arrays become complex64, and
     * float64, integer and bool arrays become complex128.
     *
     * @param array the array's dtype
     * @return the promoted dtype
     */
    public static DType resultTypeWithComplex(DType array) {
        if (array.isComplex()) {
            return array;
        }
        return array.isFloatingPoint() && array != DType.FLOAT64 ? DType.COMPLEX64 : DType.COMPLEX128;
    }

    /**
     * Tells whether {@code from} can be converted to {@code to} with {@link Casting#SAFE}
     * casting ({@code ARRAY_API}: {@code can_cast}).
     *
     * @param from source dtype
     * @param to   target dtype
     * @return {@code true} if every value of {@code from} is representable in {@code to}
     */
    public static boolean canCast(DType from, DType to) {
        return canCast(from, to, Casting.SAFE);
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
        Objects.requireNonNull(from, "from");
        Objects.requireNonNull(to, "to");
        return switch (Objects.requireNonNull(casting, "casting")) {
            case NO, EQUIV -> from == to;
            case SAFE -> resultType(from, to) == to;
            case SAME_KIND -> resultType(from, to) == to || kindOrder(to) >= kindOrder(from);
            case UNSAFE -> true;
        };
    }

    /**
     * bool &lt; unsigned &lt; signed &lt; float &lt; complex.
     */
    private static int kindOrder(DType t) {
        return switch (t.kind()) {
            case BOOL -> 0;
            case UNSIGNED_INTEGER -> 1;
            case SIGNED_INTEGER -> 2;
            case FLOATING -> 3;
            case COMPLEX -> 4;
        };
    }

    private static DType widerFloat(DType a, DType b) {
        boolean ha = a == DType.FLOAT16 || a == DType.BFLOAT16;
        boolean hb = b == DType.FLOAT16 || b == DType.BFLOAT16;
        if (ha && hb) {
            return DType.FLOAT32;  // float16 + bfloat16 (equal dtypes were handled before)
        }
        return a.size() >= b.size() ? a : b;
    }

    /**
     * The real dtype of one component of a complex dtype.
     *
     * @param complex complex64 or complex128
     * @return float32 or float64
     */
    public static DType component(DType complex) {
        return switch (complex) {
            case COMPLEX64 -> DType.FLOAT32;
            case COMPLEX128 -> DType.FLOAT64;
            default -> throw new DTypeException(complex.typeName() + " is not a complex dtype");
        };
    }
}
