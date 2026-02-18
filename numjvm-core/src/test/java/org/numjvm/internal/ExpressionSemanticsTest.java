package org.numjvm.internal;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.Ufuncs;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ExpressionSemanticsTest {
    @Test
    void resolutionMatchesEagerFloatingOperations() {
        for (DType left : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
            var array = NJ.ones(left, 2);
            assertEquals(Ufuncs.SQRT.apply(array).dtype(), ExpressionSemantics.unaryResult(Ufuncs.SQRT, left));
            assertEquals(Ufuncs.ADD.apply(array, 1.25).dtype(),
                    ExpressionSemantics.weakDoubleResult(Ufuncs.ADD, left));
            for (DType right : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
                assertEquals(Ufuncs.MULTIPLY.apply(array, NJ.ones(right, 2)).dtype(),
                        ExpressionSemantics.binaryResult(Ufuncs.MULTIPLY, left, right));
            }
        }
    }
}
