package org.numjvm.jit;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/** Memory accessors linked only from generated classes in this module. */
final class GeneratedAccess {
    private GeneratedAccess() { }
    static double readFloat(MemorySegment segment, long offset) { return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset); }
    static double readDouble(MemorySegment segment, long offset) { return segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset); }
    static void writeFloat(MemorySegment segment, long offset, double value) { segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, (float) value); }
    static void writeDouble(MemorySegment segment, long offset, double value) { segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset, value); }
}
