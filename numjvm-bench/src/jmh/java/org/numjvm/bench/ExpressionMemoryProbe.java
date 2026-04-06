package org.numjvm.bench;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.jit.Jit;

/** Isolated scoped accounting; heap GC metrics are collected separately by JMH. */
public final class ExpressionMemoryProbe {
    private ExpressionMemoryProbe() { }
    public static void main(String[] args) {
        var initial = NJ.memoryStats();
        try (var scope = NJ.scope(); var context = Jit.context(Jit.Mode.BLOCK)) {
            var input = scope.ones(DType.FLOAT64, 1_048_576);
            var output = scope.zeros(DType.FLOAT64, 1_048_576);
            var expression = context.trace(1, i -> i.input(0).square().add(1));
            var before = NJ.memoryStats();
            expression.evalInto(output, input);
            var after = NJ.memoryStats();
            System.out.println("scopedInputOutputBytes=" + (before.liveNativeBytes() - initial.liveNativeBytes()));
            System.out.println("retainedNativeBytesAfterExecution=" + (after.liveNativeBytes() - before.liveNativeBytes()));
            System.out.println("nativeAllocationsDuringExecution=" + (after.nativeAllocations() - before.nativeAllocations()));
            System.out.println("processPeakNativeBytes=" + after.peakNativeBytes());
            if (output.getDouble(0) != 2) throw new AssertionError("incorrect output");
        }
        System.out.println("retainedNativeBytesAfterScopeClose=" + (NJ.memoryStats().liveNativeBytes() - initial.liveNativeBytes()));
    }
}
