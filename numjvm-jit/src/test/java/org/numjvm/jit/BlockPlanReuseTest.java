package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;

class BlockPlanReuseTest {
    @Test void reusesTypedScheduleWithoutRetainingInputs() {
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var expression = context.trace(2, i -> {
                var x = i.input(0);
                return x.multiply(x).add(i.input(1));
            });
            var floats = new DType[]{DType.FLOAT32, DType.FLOAT32};
            var mixed = new DType[]{DType.FLOAT32, DType.FLOAT64};
            var plan = expression.blockPlan(floats);
            assertSame(plan, expression.blockPlan(floats.clone()));
            assertNotSame(plan, expression.blockPlan(mixed));
            var result = expression.eval(NJ.ones(DType.FLOAT32, 4), NJ.ones(DType.FLOAT64, 4));
            assertEquals(DType.FLOAT64, result.dtype());
            assertEquals(2, result.getDouble(0));
            assertSame(plan, expression.blockPlan(floats));
        }
    }

    @Test void specializationCacheEvictsOldSignatures() {
        try (var context = Jit.context(Jit.Mode.BLOCK)) {
            var expression = context.trace(4, i -> i.input(0).add(i.input(1)).add(i.input(2)).add(i.input(3)));
            var first = new DType[]{DType.FLOAT32, DType.FLOAT32, DType.FLOAT32, DType.FLOAT32};
            var old = expression.blockPlan(first);
            for (int bits = 1; bits <= 8; bits++) {
                DType[] types = new DType[4];
                for (int i = 0; i < 4; i++) types[i] = (bits & (1 << i)) == 0 ? DType.FLOAT32 : DType.FLOAT64;
                expression.blockPlan(types);
            }
            assertNotSame(old, expression.blockPlan(first));
        }
    }
}
