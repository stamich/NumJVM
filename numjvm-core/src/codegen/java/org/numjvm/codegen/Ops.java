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
                    .build());
}
