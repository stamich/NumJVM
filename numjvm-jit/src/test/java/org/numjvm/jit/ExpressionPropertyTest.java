package org.numjvm.jit;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ExpressionPropertyTest {
    @Property(tries = 50)
    void sharedSubexpressionsMatchEager(@ForAll @IntRange(min = -2000, max = 2000) int value,
                                         @ForAll @IntRange(min = -10, max = 10) int scalar) {
        float x = value / 7f;
        var input = NJ.array(new float[]{x});
        var twice = Ufuncs.ADD.apply(input, (double) scalar);
        var expected = Ufuncs.ADD.apply(Ufuncs.MULTIPLY.apply(twice, twice), twice);
        for (Jit.Mode mode : Jit.Mode.values()) {
            try (var context = Jit.context(mode)) {
                var graph = context.trace(1, i -> {
                    var shared = i.input(0).add((double) scalar);
                    return shared.multiply(shared).add(shared);
                });
                var actual = graph.eval(input);
                assertEquals(DType.FLOAT32, actual.dtype());
                assertEquals(Float.floatToRawIntBits((float) expected.getDouble(0)),
                        Float.floatToRawIntBits((float) actual.getDouble(0)));
            }
        }
    }
}
