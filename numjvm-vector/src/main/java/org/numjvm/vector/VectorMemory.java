package org.numjvm.vector;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

/** Small helpers used by generated loops; vector values never enter an array. */
final class VectorMemory {
    private VectorMemory() { }
    static FloatVector loadFloat(MemorySegment segment, long offset, long stride) {
        var species = FloatVector.SPECIES_PREFERRED;
        return stride == 0 ? FloatVector.broadcast(species, segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset))
                : FloatVector.fromMemorySegment(species, segment, offset, ByteOrder.nativeOrder());
    }
    static FloatVector loadMixedFloat(MemorySegment segment, long offset, long stride) {
        var species = MixedConversions.FLOAT;
        return stride == 0 ? FloatVector.broadcast(species, segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset))
                : FloatVector.fromMemorySegment(species, segment, offset, ByteOrder.nativeOrder());
    }
    static DoubleVector loadDouble(MemorySegment segment, long offset, long stride) {
        var species = DoubleVector.SPECIES_PREFERRED;
        return stride == 0 ? DoubleVector.broadcast(species, segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset))
                : DoubleVector.fromMemorySegment(species, segment, offset, ByteOrder.nativeOrder());
    }
    static void storeFloat(MemorySegment segment, long offset, FloatVector value) {
        value.intoMemorySegment(segment, offset, ByteOrder.nativeOrder());
    }
    static void storeDouble(MemorySegment segment, long offset, DoubleVector value) {
        value.intoMemorySegment(segment, offset, ByteOrder.nativeOrder());
    }
}
