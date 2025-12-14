package org.numjvm.ufunc;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.numjvm.dtype.DType;

/**
 * All of NumJVM's ufuncs.
 *
 * <p>Each constant follows NumPy's ufunc of the same name; the dtype rules are those of
 * ADR-005 and the edge cases (overflow, division by zero, NaN) those of decision D5. Integer
 * arithmetic wraps on overflow.
 */
public final class Ufuncs {

    private Ufuncs() {
    }

    /**
     * {@code x + y}; logical or for bool ({@code ARRAY_API}: {@code add}).
     */
    public static final BinaryUfunc ADD = BinaryUfunc.builder("add").build();

    /**
     * {@code x - y} ({@code ARRAY_API}: {@code subtract}). Not defined for two bool operands,
     * as in NumPy; use a logical or bitwise xor instead.
     */
    public static final BinaryUfunc SUBTRACT = BinaryUfunc.builder("subtract")
            .unsupported(DType.BOOL, "boolean subtract is not supported; use LOGICAL_XOR or BITWISE_XOR instead")
            .build();

    /**
     * {@code x * y}; logical and for bool ({@code ARRAY_API}: {@code multiply}).
     */
    public static final BinaryUfunc MULTIPLY = BinaryUfunc.builder("multiply").build();

    /**
     * True division {@code x / y} ({@code ARRAY_API}: {@code divide}). Integer and bool
     * operands give float64, as in NumPy; float16 is computed in float32.
     */
    public static final BinaryUfunc DIVIDE = BinaryUfunc.builder("divide")
            .loopFor(t -> t.isInteger() || t.isBool() ? DType.FLOAT64 : null)
            .build();

    /**
     * Floor division {@code floor(x / y)} ({@code ARRAY_API}: {@code floor_divide}). Integer
     * division by zero gives 0; two bool operands give int8. Floating-point results follow
     * NumPy's {@code npy_floor_divide}.
     */
    public static final BinaryUfunc FLOOR_DIVIDE = BinaryUfunc.builder("floor_divide").build();

    /**
     * Remainder of floor division, with the sign of the divisor ({@code ARRAY_API}:
     * {@code remainder}; Python's {@code %}). Integer remainder by zero gives 0;
     * floating-point remainder by zero gives NaN.
     */
    public static final BinaryUfunc REMAINDER = BinaryUfunc.builder("remainder").build();

    /**
     * {@code x} to the power {@code y} ({@code ARRAY_API}: {@code pow}; NumPy's
     * {@code power}). Integer powers wrap on overflow; a negative integer exponent is a
     * {@link org.numjvm.exception.ValueException}. Two bool operands give int8.
     * Floating-point special cases follow C99 ({@code pow(1, NaN) == 1}).
     */
    public static final BinaryUfunc POWER = BinaryUfunc.builder("power").build();

    /**
     * The larger operand, element-wise; NaN propagates ({@code ARRAY_API}: {@code maximum}).
     */
    public static final BinaryUfunc MAXIMUM = BinaryUfunc.builder("maximum").build();

    /**
     * The smaller operand, element-wise; NaN propagates ({@code ARRAY_API}: {@code minimum}).
     */
    public static final BinaryUfunc MINIMUM = BinaryUfunc.builder("minimum").build();

    /**
     * The larger operand, ignoring a NaN operand ({@code NUMPY_COMPATIBILITY}: {@code fmax}).
     */
    public static final BinaryUfunc FMAX = BinaryUfunc.builder("fmax").build();

    /**
     * The smaller operand, ignoring a NaN operand ({@code NUMPY_COMPATIBILITY}: {@code fmin}).
     */
    public static final BinaryUfunc FMIN = BinaryUfunc.builder("fmin").build();

    /**
     * {@code x == y}, giving bool ({@code ARRAY_API}: {@code equal}). Comparisons follow IEEE 754
     * (NaN is unequal to everything); signed integers against uint64, and integers against a
     * Java {@code long} outside their range, are compared exactly ({@code NUMPY_COMPATIBILITY},
     * as NumPy 2).
     */
    public static final BinaryUfunc EQUAL = BinaryUfunc.builder("equal").comparison().build();

    /**
     * {@code x != y}, giving bool ({@code ARRAY_API}: {@code not_equal}); see {@link #EQUAL}.
     */
    public static final BinaryUfunc NOT_EQUAL = BinaryUfunc.builder("not_equal").comparison().build();

    /**
     * {@code x < y}, giving bool ({@code ARRAY_API}: {@code less}); see {@link #EQUAL}.
     */
    public static final BinaryUfunc LESS = BinaryUfunc.builder("less").comparison().build();

    /**
     * {@code x <= y}, giving bool ({@code ARRAY_API}: {@code less_equal}); see {@link #EQUAL}.
     */
    public static final BinaryUfunc LESS_EQUAL = BinaryUfunc.builder("less_equal").comparison().build();

    /**
     * {@code x > y}, giving bool ({@code ARRAY_API}: {@code greater}); see {@link #EQUAL}.
     */
    public static final BinaryUfunc GREATER = BinaryUfunc.builder("greater").comparison().build();

    /**
     * {@code x >= y}, giving bool ({@code ARRAY_API}: {@code greater_equal}); see {@link #EQUAL}.
     */
    public static final BinaryUfunc GREATER_EQUAL = BinaryUfunc.builder("greater_equal").comparison().build();

    /**
     * Logical and, giving bool ({@code ARRAY_API}: {@code logical_and}). Operands of any real
     * dtype are true when non-zero (NaN included).
     */
    public static final BinaryUfunc LOGICAL_AND = BinaryUfunc.builder("logical_and").loopFor(t -> DType.BOOL).build();

    /**
     * Logical or, giving bool ({@code ARRAY_API}: {@code logical_or}); see {@link #LOGICAL_AND}.
     */
    public static final BinaryUfunc LOGICAL_OR = BinaryUfunc.builder("logical_or").loopFor(t -> DType.BOOL).build();

    /**
     * Logical exclusive or, giving bool ({@code ARRAY_API}: {@code logical_xor}); see
     * {@link #LOGICAL_AND}.
     */
    public static final BinaryUfunc LOGICAL_XOR = BinaryUfunc.builder("logical_xor").loopFor(t -> DType.BOOL).build();

    /**
     * Logical not, giving bool ({@code ARRAY_API}: {@code logical_not}); see {@link #LOGICAL_AND}.
     */
    public static final UnaryUfunc LOGICAL_NOT = UnaryUfunc.builder("logical_not").loopFor(t -> DType.BOOL).build();

    /**
     * Bitwise and of bool and integer operands ({@code ARRAY_API}: {@code bitwise_and});
     * floating-point operands are a {@link org.numjvm.exception.DTypeException}.
     */
    public static final BinaryUfunc BITWISE_AND = BinaryUfunc.builder("bitwise_and").build();

    /**
     * Bitwise or of bool and integer operands ({@code ARRAY_API}: {@code bitwise_or}).
     */
    public static final BinaryUfunc BITWISE_OR = BinaryUfunc.builder("bitwise_or").build();

    /**
     * Bitwise exclusive or of bool and integer operands ({@code ARRAY_API}: {@code bitwise_xor}).
     */
    public static final BinaryUfunc BITWISE_XOR = BinaryUfunc.builder("bitwise_xor").build();

    /**
     * Bitwise not of integers, logical not of bool ({@code ARRAY_API}: {@code bitwise_invert};
     * NumPy's {@code invert}).
     */
    public static final UnaryUfunc INVERT = UnaryUfunc.builder("invert").build();

    /**
     * {@code -x} ({@code ARRAY_API}: {@code negative}). Integers wrap ({@code -MIN == MIN}, the
     * negative of an unsigned value wraps); bool is not supported, as in NumPy.
     */
    public static final UnaryUfunc NEGATIVE = UnaryUfunc.builder("negative")
            .unsupported(DType.BOOL, "boolean negative is not supported; use INVERT or LOGICAL_NOT instead")
            .build();

    /**
     * {@code +x}, a copy ({@code ARRAY_API}: {@code positive}); bool is not supported, as in NumPy.
     */
    public static final UnaryUfunc POSITIVE = UnaryUfunc.builder("positive")
            .unsupported(DType.BOOL, "ufunc 'positive' has no loop for (bool)")
            .build();

    /**
     * {@code |x|} ({@code ARRAY_API}: {@code abs}; NumPy's {@code absolute}). {@code abs(MIN)}
     * wraps to {@code MIN}; {@code abs(-0.0) == 0.0}; bool and unsigned values are unchanged.
     */
    public static final UnaryUfunc ABSOLUTE = UnaryUfunc.builder("absolute").build();

    /**
     * -1, 0 or 1 by the sign of {@code x} ({@code ARRAY_API}: {@code sign}); NaN gives NaN and
     * {@code -0.0} gives {@code 0.0}. Bool is not supported, as in NumPy.
     */
    public static final UnaryUfunc SIGN = UnaryUfunc.builder("sign")
            .unsupported(DType.BOOL, "ufunc 'sign' has no loop for (bool)")
            .build();

    /**
     * {@code x * x} ({@code ARRAY_API}: {@code square}). Integers wrap; bool gives int8.
     */
    public static final UnaryUfunc SQUARE = UnaryUfunc.builder("square").build();

    /**
     * The square root, correctly rounded ({@code ARRAY_API}: {@code sqrt}).
     * Exact, as IEEE 754 requires.
     * Integer and bool inputs give the first float dtype they convert to safely: float16
     * for bool, int8 and uint8, float32 for 16-bit integers, float64 otherwise.
     */
    public static final UnaryUfunc SQRT = UnaryUfunc.builder("sqrt").build();

    /**
     * E to the power {@code x} ({@code ARRAY_API}: {@code exp}).
     * Computed with {@link Math#exp(double)}; results may differ from NumPy by up to 4 ulp (decision D6).
     * Integer and bool inputs give the first float dtype they convert to safely: float16
     * for bool, int8 and uint8, float32 for 16-bit integers, float64 otherwise.
     */
    public static final UnaryUfunc EXP = UnaryUfunc.builder("exp").build();

    /**
     * The natural logarithm; {@code log(0) == -inf}, negative values give nan ({@code ARRAY_API}: {@code log}).
     * Computed with {@link Math#log(double)}; results may differ from NumPy by up to 4 ulp (decision D6).
     * Integer and bool inputs give the first float dtype they convert to safely: float16
     * for bool, int8 and uint8, float32 for 16-bit integers, float64 otherwise.
     */
    public static final UnaryUfunc LOG = UnaryUfunc.builder("log").build();

    /**
     * The sine of {@code x} in radians ({@code ARRAY_API}: {@code sin}).
     * Computed with {@link Math#sin(double)}; results may differ from NumPy by up to 4 ulp (decision D6).
     * Integer and bool inputs give the first float dtype they convert to safely: float16
     * for bool, int8 and uint8, float32 for 16-bit integers, float64 otherwise.
     */
    public static final UnaryUfunc SIN = UnaryUfunc.builder("sin").build();

    /**
     * The cosine of {@code x} in radians ({@code ARRAY_API}: {@code cos}).
     * Computed with {@link Math#cos(double)}; results may differ from NumPy by up to 4 ulp (decision D6).
     * Integer and bool inputs give the first float dtype they convert to safely: float16
     * for bool, int8 and uint8, float32 for 16-bit integers, float64 otherwise.
     */
    public static final UnaryUfunc COS = UnaryUfunc.builder("cos").build();

    /**
     * The hyperbolic tangent ({@code ARRAY_API}: {@code tanh}).
     * Computed with {@link Math#tanh(double)}; results may differ from NumPy by up to 4 ulp (decision D6).
     * Integer and bool inputs give the first float dtype they convert to safely: float16
     * for bool, int8 and uint8, float32 for 16-bit integers, float64 otherwise.
     */
    public static final UnaryUfunc TANH = UnaryUfunc.builder("tanh").build();

    /**
     * Returns every ufunc, in declaration order.
     *
     * @return the ufuncs
     */
    public static List<Ufunc> all() {
        return List.copyOf(ByName.MAP.values());
    }

    /**
     * Looks a ufunc up by its NumPy name, e.g. {@code "add"}.
     *
     * @param name the name
     * @return the ufunc, or empty if NumJVM has no ufunc of that name
     */
    public static Optional<Ufunc> byName(String name) {
        return Optional.ofNullable(ByName.MAP.get(name));
    }

    /**
     * Built on first use, after all constants are initialised. Every public constant must be
     * listed here (UfuncsTest checks it).
     */
    private static final class ByName {
        static final Map<String, Ufunc> MAP = build(List.of(
                ADD, SUBTRACT, MULTIPLY, DIVIDE, FLOOR_DIVIDE, REMAINDER, POWER, MAXIMUM, MINIMUM, FMAX, FMIN,
                EQUAL, NOT_EQUAL, LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR,
                LOGICAL_NOT, BITWISE_AND, BITWISE_OR, BITWISE_XOR, INVERT, NEGATIVE, POSITIVE, ABSOLUTE, SIGN,
                SQUARE, SQRT, EXP, LOG, SIN, COS, TANH));

        private static Map<String, Ufunc> build(List<Ufunc> all) {
            Map<String, Ufunc> m = new LinkedHashMap<>();
            for (Ufunc u : all) {
                m.put(u.name(), u);
            }
            return java.util.Collections.unmodifiableMap(m);
        }
    }
}
