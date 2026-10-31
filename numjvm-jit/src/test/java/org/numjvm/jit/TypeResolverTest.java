package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;

class TypeResolverTest {
    @Test void resolvesIntermediateTypesAndWeakConstants() {
        try (var context = Jit.context()) {
            Expr[] intermediate = new Expr[1];
            var expression = context.trace(2, i -> {
                intermediate[0] = i.input(0).add(1.0).square();
                return intermediate[0].multiply(i.input(1));
            });
            var resolver = new TypeResolver(new DType[]{DType.FLOAT32, DType.FLOAT64});
            assertEquals(DType.FLOAT32, resolver.resolve(intermediate[0]));
            assertEquals(DType.FLOAT64, resolver.resolve(expression.root()));
            assertThrows(IllegalArgumentException.class, () -> new TypeResolver(new DType[]{DType.INT32}));
        }
    }
}
