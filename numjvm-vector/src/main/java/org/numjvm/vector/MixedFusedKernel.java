package org.numjvm.vector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.List;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.FusedKernel;

/** Equal-lane mixed-dtype traversal; scalar tails use the same generated graph. */
final class MixedFusedKernel implements FusedKernel {
    private final MixedExpression expression;
    private final List<DType> inputTypes;
    MixedFusedKernel(MixedExpression expression, List<DType> inputTypes) {
        this.expression = expression; this.inputTypes = List.copyOf(inputTypes);
    }
    @Override public void run(MemorySegment[] segments, long[] offsets, long[] strides, long count) {
        var doubles = MixedConversions.DOUBLE;
        var floats = MixedConversions.FLOAT;
        Object[] values = new Object[inputTypes.size()];
        long bound = doubles.loopBound(count), i = 0;
        for (; i < bound; i += doubles.length()) {
            for (int slot = 0; slot < values.length; slot++) {
                if (inputTypes.get(slot) == DType.FLOAT32) {
                    values[slot] = strides[slot] == 0
                            ? FloatVector.broadcast(floats, segments[slot].get(ValueLayout.JAVA_FLOAT_UNALIGNED, offsets[slot]))
                            : FloatVector.fromMemorySegment(floats, segments[slot], offsets[slot] + i * strides[slot], ByteOrder.nativeOrder());
                } else {
                    values[slot] = strides[slot] == 0
                            ? DoubleVector.broadcast(doubles, segments[slot].get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[slot]))
                            : DoubleVector.fromMemorySegment(doubles, segments[slot], offsets[slot] + i * strides[slot], ByteOrder.nativeOrder());
                }
            }
            expression.evaluate(values).intoMemorySegment(segments[values.length], offsets[values.length] + i * strides[values.length], ByteOrder.nativeOrder());
        }
        for (; i < count; i++) {
            for (int slot = 0; slot < values.length; slot++) {
                if (inputTypes.get(slot) == DType.FLOAT32)
                    values[slot] = FloatVector.broadcast(floats, segments[slot].get(ValueLayout.JAVA_FLOAT_UNALIGNED, offsets[slot] + i * strides[slot]));
                else values[slot] = DoubleVector.broadcast(doubles, segments[slot].get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[slot] + i * strides[slot]));
            }
            segments[values.length].set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offsets[values.length] + i * strides[values.length], expression.evaluate(values).lane(0));
        }
    }
}
