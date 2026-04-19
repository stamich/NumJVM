package org.numjvm.vector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import org.numjvm.internal.backend.FusedKernel;

/** Traverses contiguous runs and delegates each vector expression to generated code. */
final class DoubleFusedKernel implements FusedKernel {
    private final DoubleExpression expression;
    DoubleFusedKernel(DoubleExpression expression) { this.expression = expression; }

    @Override public void run(MemorySegment[] segments, long[] offsets, long[] strides, long count) {
        var species = DoubleVector.SPECIES_PREFERRED;
        var values = new DoubleVector[segments.length - 1];
        long bound = species.loopBound(count), i = 0;
        for (; i < bound; i += species.length()) {
            for (int slot = 0; slot < values.length; slot++)
                values[slot] = strides[slot] == 0
                        ? DoubleVector.broadcast(species, segments[slot].get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[slot]))
                        : DoubleVector.fromMemorySegment(species, segments[slot], offsets[slot] + i * strides[slot], ByteOrder.nativeOrder());
            expression.evaluate(values).intoMemorySegment(segments[values.length], offsets[values.length] + i * strides[values.length], ByteOrder.nativeOrder());
        }
        // Scalar loads/stores keep short tails within their memory segment bounds.
        for (; i < count; i++) {
            for (int slot = 0; slot < values.length; slot++)
                values[slot] = DoubleVector.broadcast(species, segments[slot].get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[slot] + i * strides[slot]));
            double value = expression.evaluate(values).lane(0);
            segments[values.length].set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[values.length] + i * strides[values.length], value);
        }
    }
}
