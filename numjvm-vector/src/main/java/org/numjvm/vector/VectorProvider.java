package org.numjvm.vector;

import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.Accelerators;
import org.numjvm.internal.backend.LoopProvider;
import org.numjvm.internal.loops.BinaryLoop;

/** Optional service implementation; every unsupported run delegates to its scalar loop. */
public final class VectorProvider implements LoopProvider {
    /** Constructs the service provider. */
    public VectorProvider() { }
    @Override public String name() { return "jdk-vector"; }
    @Override public BinaryLoop binary(String op, DType type, BinaryLoop scalar) {
        if (!op.equals("add") || type != DType.FLOAT64) { return scalar; }
        return (a, pa, sa, b, pb, sb, out, po, so, n) -> {
            var species = DoubleVector.SPECIES_PREFERRED;
            long bound = species.loopBound(n);
            if (sa != 8 || sb != 8 || so != 8 || bound == 0) {
                scalar.run(a, pa, sa, b, pb, sb, out, po, so, n); return;
            }
            for (long i = 0; i < bound; i += species.length()) {
                var x = DoubleVector.fromMemorySegment(species, a, pa + i * 8, ByteOrder.nativeOrder());
                var y = DoubleVector.fromMemorySegment(species, b, pb + i * 8, ByteOrder.nativeOrder());
                x.add(y).intoMemorySegment(out, po + i * 8, ByteOrder.nativeOrder());
            }
            scalar.run(a, pa + bound * sa, sa, b, pb + bound * sb, sb, out, po + bound * so, so, n - bound);
            Accelerators.executed();
        };
    }
}
