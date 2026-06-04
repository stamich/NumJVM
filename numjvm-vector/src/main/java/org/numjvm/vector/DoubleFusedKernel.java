package org.numjvm.vector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import org.numjvm.internal.backend.FusedKernel;
import org.numjvm.internal.backend.Accelerators;

/** Traverses contiguous runs and delegates each vector expression to generated code. */
final class DoubleFusedKernel implements FusedKernel {
    private final DoubleExpression expression;
    DoubleFusedKernel(DoubleExpression expression) { this.expression = expression; }
    @Override public int vectorLength() { return DoubleVector.SPECIES_PREFERRED.length(); }

    @Override public void run(MemorySegment[] segments, long[] offsets, long[] strides, long count) {
        var species = DoubleVector.SPECIES_PREFERRED;
        var values = new DoubleVector[segments.length - 1];
        long bound = species.loopBound(count), i = 0;
        if (bound > 0) {
            ((VectorLoop) expression).runVectors(segments, offsets, strides, bound, species.length());
            i = bound;
        }
        // Scalar loads/stores keep short tails within their memory segment bounds.
        for (; i < count; i++) {
            for (int slot = 0; slot < values.length; slot++)
                values[slot] = DoubleVector.broadcast(species, segments[slot].get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[slot] + i * strides[slot]));
            double value = expression.evaluate(values).lane(0);
            segments[values.length].set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[values.length] + i * strides[values.length], value);
        }
        if (bound > 0) Accelerators.executed();
    }
}
