package org.numjvm.vector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import org.numjvm.internal.backend.FusedKernel;

/** Traverses contiguous runs and delegates each vector expression to generated code. */
final class FloatFusedKernel implements FusedKernel {
    private final FloatExpression expression;
    FloatFusedKernel(FloatExpression expression) { this.expression = expression; }

    @Override public void run(MemorySegment[] segments, long[] offsets, long[] strides, long count) {
        var species = FloatVector.SPECIES_PREFERRED;
        var values = new FloatVector[segments.length - 1];
        long bound = species.loopBound(count), i = 0;
        for (; i < bound; i += species.length()) {
            for (int slot = 0; slot < values.length; slot++)
                values[slot] = FloatVector.fromMemorySegment(species, segments[slot], offsets[slot] + i * strides[slot], ByteOrder.nativeOrder());
            expression.evaluate(values).intoMemorySegment(segments[values.length], offsets[values.length] + i * strides[values.length], ByteOrder.nativeOrder());
        }
        // Scalar loads/stores keep short tails within their memory segment bounds.
        for (; i < count; i++) {
            for (int slot = 0; slot < values.length; slot++)
                values[slot] = FloatVector.broadcast(species, segments[slot].get(ValueLayout.JAVA_FLOAT_UNALIGNED, offsets[slot] + i * strides[slot]));
            float value = expression.evaluate(values).lane(0);
            segments[values.length].set(ValueLayout.JAVA_FLOAT_UNALIGNED, offsets[values.length] + i * strides[values.length], value);
        }
    }
}
