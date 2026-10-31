package org.numjvm.jit;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.ShapeException;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ufunc.Ufuncs;
import static org.junit.jupiter.api.Assertions.*;

class LayoutResolverTest {
    @Test void matchesEagerIntermediateLayout() {
        NDArray a = NJ.ones(DType.FLOAT32, 3, 2).transpose();
        NDArray b = NJ.ones(DType.FLOAT32, 3, 2).transpose();
        try (var context = Jit.context()) {
            var graph = context.trace(2, i -> i.input(0).add(i.input(1)).square());
            var actual = new LayoutResolver(new NDArray[]{a, b}, new TypeResolver(new DType[]{a.dtype(), b.dtype()})).resolve(graph.root());
            NDArray eager = Ufuncs.SQUARE.apply(Ufuncs.ADD.apply(a, b));
            assertEquals(eager.shape(), actual.shape());
            assertArrayEquals(eager.strides().values(), actual.strides().values());
        }
    }

    @Test void rejectsInvalidIntermediateBroadcast() {
        NDArray a = NJ.ones(DType.FLOAT32, 2, 3);
        NDArray b = NJ.ones(DType.FLOAT32, 4, 3);
        try (var context = Jit.context()) {
            var graph = context.trace(2, i -> i.input(0).add(i.input(1)).square());
            var resolver = new LayoutResolver(new NDArray[]{a, b}, new TypeResolver(new DType[]{a.dtype(), b.dtype()}));
            assertThrows(ShapeException.class, () -> resolver.resolve(graph.root()));
        }
    }
}
