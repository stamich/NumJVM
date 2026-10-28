package org.numjvm.codegen;

import static org.numjvm.codegen.DTypeSpec.Family.BOOL;
import static org.numjvm.codegen.DTypeSpec.Family.DOUBLE;
import static org.numjvm.codegen.DTypeSpec.Family.FLOAT;
import static org.numjvm.codegen.DTypeSpec.Family.INT;
import static org.numjvm.codegen.DTypeSpec.Family.LONG;
import static org.numjvm.codegen.DTypeSpec.Family.ULONG;

import java.util.List;

/**
 * The table of generated element-wise operations. Adding an operation here and
 * rebuilding generates its loops for every dtype family that has an expression, and
 * registers them in the kernel registry.
 */
final class Ops {

    private Ops() {
    }

    static final List<OpSpec> ALL = List.of(
            OpSpec.binary("add", "x + y (logical or for bool)")
                    .on("x | y", BOOL)
                    .on("x + y", INT, LONG, ULONG, FLOAT, DOUBLE)
                    .build(),
            OpSpec.binary("subtract", "x - y")
                    .on("x - y", INT, LONG, ULONG, FLOAT, DOUBLE)
                    .build(),
            OpSpec.binary("multiply", "x * y (logical and for bool)")
                    .on("x & y", BOOL)
                    .on("x * y", INT, LONG, ULONG, FLOAT, DOUBLE)
                    .build(),
            // true division: integer and bool operands are converted to float64 by the ufunc
            OpSpec.binary("divide", "x / y, true division")
                    .on("x / y", FLOAT, DOUBLE)
                    .build(),
            OpSpec.binary("floor_divide", "floor(x / y); 0 for integers divided by zero")
                    .on("ScalarMath.floorDivide(x, y)", INT, LONG, FLOAT, DOUBLE)
                    .on("ScalarMath.floorDivideUnsigned(x, y)", ULONG)
                    .build(),
            OpSpec.binary("remainder", "x - floor(x / y) * y, with the sign of y; 0 for integers modulo zero")
                    .on("ScalarMath.remainder(x, y)", INT, LONG, FLOAT, DOUBLE)
                    .on("ScalarMath.remainderUnsigned(x, y)", ULONG)
                    .build(),
            OpSpec.binary("power", "x to the power y; integers wrap, a negative integer exponent is an error")
                    .on("ScalarMath.power(x, y)", INT, LONG, FLOAT, DOUBLE)
                    .on("ScalarMath.powerUnsigned(x, y)", ULONG)
                    .build(),
            OpSpec.binary("maximum", "the larger of x and y; NaN propagates")
                    .on("x | y", BOOL)
                    .on("Math.max(x, y)", INT, LONG)
                    .on("Long.compareUnsigned(x, y) >= 0 ? x : y", ULONG)
                    .on("ScalarMath.maximum(x, y)", FLOAT, DOUBLE)
                    // NumPy's half loop keeps x when x == y: maximum(0.0, -0.0) is 0.0
                    .onDType("ScalarMath.maximumHalf(x, y)", "FLOAT16")
                    .build(),
            OpSpec.binary("minimum", "the smaller of x and y; NaN propagates")
                    .on("x & y", BOOL)
                    .on("Math.min(x, y)", INT, LONG)
                    .on("Long.compareUnsigned(x, y) <= 0 ? x : y", ULONG)
                    .on("ScalarMath.minimum(x, y)", FLOAT, DOUBLE)
                    .onDType("ScalarMath.minimumHalf(x, y)", "FLOAT16")
                    .build(),
            OpSpec.binary("fmax", "the larger of x and y; a NaN operand is ignored")
                    .on("x | y", BOOL)
                    .on("Math.max(x, y)", INT, LONG)
                    .on("Long.compareUnsigned(x, y) >= 0 ? x : y", ULONG)
                    .on("ScalarMath.fmax(x, y)", FLOAT, DOUBLE)
                    .build(),
            OpSpec.binary("fmin", "the smaller of x and y; a NaN operand is ignored")
                    .on("x & y", BOOL)
                    .on("Math.min(x, y)", INT, LONG)
                    .on("Long.compareUnsigned(x, y) <= 0 ? x : y", ULONG)
                    .on("ScalarMath.fmin(x, y)", FLOAT, DOUBLE)
                    .build(),
            // comparisons: bool output; NaN compares unequal to everything (IEEE 754)
            OpSpec.binary("equal", "x == y").boolOutput()
                    .on("x == y", BOOL, INT, LONG, ULONG, FLOAT, DOUBLE)
                    .build(),
            OpSpec.binary("not_equal", "x != y").boolOutput()
                    .on("x != y", BOOL, INT, LONG, ULONG, FLOAT, DOUBLE)
                    .build(),
            OpSpec.binary("less", "x < y").boolOutput()
                    .on("!x & y", BOOL)
                    .on("x < y", INT, LONG, FLOAT, DOUBLE)
                    .on("Long.compareUnsigned(x, y) < 0", ULONG)
                    .build(),
            OpSpec.binary("less_equal", "x <= y").boolOutput()
                    .on("!x | y", BOOL)
                    .on("x <= y", INT, LONG, FLOAT, DOUBLE)
                    .on("Long.compareUnsigned(x, y) <= 0", ULONG)
                    .build(),
            OpSpec.binary("greater", "x > y").boolOutput()
                    .on("x & !y", BOOL)
                    .on("x > y", INT, LONG, FLOAT, DOUBLE)
                    .on("Long.compareUnsigned(x, y) > 0", ULONG)
                    .build(),
            OpSpec.binary("greater_equal", "x >= y").boolOutput()
                    .on("x | !y", BOOL)
                    .on("x >= y", INT, LONG, FLOAT, DOUBLE)
                    .on("Long.compareUnsigned(x, y) >= 0", ULONG)
                    .build(),
            // logical operations run on bool; the ufunc converts other operands with != 0
            OpSpec.binary("logical_and", "x and y").on("x & y", BOOL).build(),
            OpSpec.binary("logical_or", "x or y").on("x | y", BOOL).build(),
            OpSpec.binary("logical_xor", "x xor y").on("x ^ y", BOOL).build(),
            OpSpec.unary("logical_not", "not x").on("!x", BOOL).build(),
            // bitwise operations: bool and integers only
            OpSpec.binary("bitwise_and", "x & y").on("x & y", BOOL, INT, LONG, ULONG).build(),
            OpSpec.binary("bitwise_or", "x | y").on("x | y", BOOL, INT, LONG, ULONG).build(),
            OpSpec.binary("bitwise_xor", "x ^ y").on("x ^ y", BOOL, INT, LONG, ULONG).build(),
            OpSpec.unary("invert", "~x; logical not for bool")
                    .on("!x", BOOL)
                    .on("~x", INT, LONG, ULONG)
                    .build(),
            // unary arithmetic; integers wrap (abs(MIN) == MIN, -x of unsigned wraps)
            OpSpec.unary("negative", "-x").on("-x", INT, LONG, ULONG, FLOAT, DOUBLE).build(),
            OpSpec.unary("positive", "+x, a copy").on("x", INT, LONG, ULONG, FLOAT, DOUBLE).build(),
            OpSpec.unary("absolute", "|x|; clears the sign of -0.0")
                    .on("x", BOOL, ULONG)
                    .on("Math.abs(x)", INT, LONG, FLOAT, DOUBLE)
                    .build(),
            // sign: -1, 0 or 1; NaN stays NaN and -0.0 gives 0.0 (NumPy, unlike Math.signum)
            OpSpec.unary("sign", "-1, 0 or 1; NaN for NaN")
                    .on("Integer.signum(x)", INT)
                    .on("(long) Long.signum(x)", LONG)
                    .on("x == 0 ? 0L : 1L", ULONG)
                    .on("x > 0 ? 1f : x < 0 ? -1f : x == 0 ? 0f : x", FLOAT)
                    .on("x > 0 ? 1.0 : x < 0 ? -1.0 : x == 0 ? 0.0 : x", DOUBLE)
                    .build(),
            OpSpec.unary("square", "x * x").on("x * x", INT, LONG, ULONG, FLOAT, DOUBLE).build(),
            // floating-point functions (D6): sqrt is correctly rounded; the others use
            // java.lang.Math and may differ from NumPy by a few ulp. float16, bfloat16 and
            // float32 are computed in double and rounded once; integer and bool inputs are
            // converted by the ufunc to the first float loop they cast to safely.
            OpSpec.unary("sqrt", "square root, correctly rounded")
                    .on("(float) Math.sqrt(x)", FLOAT).on("Math.sqrt(x)", DOUBLE).build(),
            OpSpec.unary("exp", "e to the power x")
                    .on("(float) Math.exp(x)", FLOAT).on("Math.exp(x)", DOUBLE).build(),
            OpSpec.unary("log", "natural logarithm")
                    .on("(float) Math.log(x)", FLOAT).on("Math.log(x)", DOUBLE).build(),
            OpSpec.unary("sin", "sine")
                    .on("(float) Math.sin(x)", FLOAT).on("Math.sin(x)", DOUBLE).build(),
            OpSpec.unary("cos", "cosine")
                    .on("(float) Math.cos(x)", FLOAT).on("Math.cos(x)", DOUBLE).build(),
            OpSpec.unary("tanh", "hyperbolic tangent")
                    .on("(float) Math.tanh(x)", FLOAT).on("Math.tanh(x)", DOUBLE).build(),
            // rounding; integers and bool are already integral and keep their dtype
            OpSpec.unary("floor", "the largest integer <= x")
                    .on("x", BOOL, INT, LONG, ULONG)
                    .on("(float) Math.floor(x)", FLOAT).on("Math.floor(x)", DOUBLE).build(),
            OpSpec.unary("ceil", "the smallest integer >= x")
                    .on("x", BOOL, INT, LONG, ULONG)
                    .on("(float) Math.ceil(x)", FLOAT).on("Math.ceil(x)", DOUBLE).build(),
            OpSpec.unary("trunc", "x rounded toward zero")
                    .on("x", BOOL, INT, LONG, ULONG)
                    .on("(float) (x < 0 ? Math.ceil(x) : Math.floor(x))", FLOAT)
                    .on("x < 0 ? Math.ceil(x) : Math.floor(x)", DOUBLE).build(),
            OpSpec.unary("rint", "x rounded to the nearest integer, halves to even")
                    .on("(float) Math.rint(x)", FLOAT).on("Math.rint(x)", DOUBLE).build());
}
