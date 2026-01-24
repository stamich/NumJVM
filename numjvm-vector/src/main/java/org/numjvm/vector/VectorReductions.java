package org.numjvm.vector;

import java.nio.ByteOrder;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.Accelerators;
import org.numjvm.internal.loops.BinaryLoop;

/** Modular integer horizontal sums; floating extrema retain scalar tie ordering. */
final class VectorReductions {
    private VectorReductions() { }
    static BinaryLoop integerSum(DType type, BinaryLoop fallback) {
        return (a,pa,sa,b,pb,sb,out,po,so,n) -> {
            if(sa!=0 || so!=0 || a!=out || pa!=po || sb!=type.size()) {
                fallback.run(a,pa,sa,b,pb,sb,out,po,so,n);return;
            }
            long i=0;
            if(type==DType.INT32) {
                var species=IntVector.SPECIES_PREFERRED;
                var sum=IntVector.zero(species);
                long bound=species.loopBound(n);
                for(;i<bound;i+=species.length()) sum=sum.add(IntVector.fromMemorySegment(species,b,pb+i*4,ByteOrder.nativeOrder()));
                int value=a.get(ValueLayout.JAVA_INT_UNALIGNED,pa)+sum.reduceLanes(VectorOperators.ADD);
                for(;i<n;i++) value+=b.get(ValueLayout.JAVA_INT_UNALIGNED,pb+i*4);
                out.set(ValueLayout.JAVA_INT_UNALIGNED,po,value);
                if(bound>0) Accelerators.executed();
            } else {
                var species=LongVector.SPECIES_PREFERRED;
                var sum=LongVector.zero(species);
                long bound=species.loopBound(n);
                for(;i<bound;i+=species.length()) sum=sum.add(LongVector.fromMemorySegment(species,b,pb+i*8,ByteOrder.nativeOrder()));
                long value=a.get(ValueLayout.JAVA_LONG_UNALIGNED,pa)+sum.reduceLanes(VectorOperators.ADD);
                for(;i<n;i++) value+=b.get(ValueLayout.JAVA_LONG_UNALIGNED,pb+i*8);
                out.set(ValueLayout.JAVA_LONG_UNALIGNED,po,value);
                if(bound>0) Accelerators.executed();
            }
        };
    }
}
