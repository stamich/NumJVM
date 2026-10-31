package org.numjvm.vector;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.Accelerators;
import org.numjvm.internal.backend.DotLoop;

/** Reduction order follows vector lanes and may differ from scalar order. */
final class VectorDot {
    private VectorDot() { }
    static DotLoop kernel(DType type) {
        if (type == DType.FLOAT32) return (a, ao, b, bo, out, oo, n) -> {
            var species = FloatVector.SPECIES_PREFERRED;
            long bound = species.loopBound(n), i = 0;
            var sum = FloatVector.zero(species);
            for (; i < bound; i += species.length()) {
                var left = FloatVector.fromMemorySegment(species, a, ao + i * 4, ByteOrder.nativeOrder());
                var right = FloatVector.fromMemorySegment(species, b, bo + i * 4, ByteOrder.nativeOrder());
                sum = sum.add(left.mul(right));
            }
            float value = sum.reduceLanes(VectorOperators.ADD);
            for (; i < n; i++) value += a.get(ValueLayout.JAVA_FLOAT_UNALIGNED, ao + i * 4)
                    * b.get(ValueLayout.JAVA_FLOAT_UNALIGNED, bo + i * 4);
            out.set(ValueLayout.JAVA_FLOAT_UNALIGNED, oo, value);
            if (bound > 0) Accelerators.executed();
        };
        if (type == DType.FLOAT64) return (a, ao, b, bo, out, oo, n) -> {
            var species = DoubleVector.SPECIES_PREFERRED;
            long bound = species.loopBound(n), i = 0;
            var sum = DoubleVector.zero(species);
            for (; i < bound; i += species.length()) {
                var left = DoubleVector.fromMemorySegment(species, a, ao + i * 8, ByteOrder.nativeOrder());
                var right = DoubleVector.fromMemorySegment(species, b, bo + i * 8, ByteOrder.nativeOrder());
                sum = sum.add(left.mul(right));
            }
            double value = sum.reduceLanes(VectorOperators.ADD);
            for (; i < n; i++) value += a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, ao + i * 8)
                    * b.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, bo + i * 8);
            out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, oo, value);
            if (bound > 0) Accelerators.executed();
        };
        return null;
    }
}
