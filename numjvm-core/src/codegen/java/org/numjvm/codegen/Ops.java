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
                    .build());
}
