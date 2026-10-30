package org.numjvm.vector;

import java.nio.ByteOrder;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.ByteVector;
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
    static BinaryLoop truth(boolean all, BinaryLoop scalar) {
        return (a,pa,sa,b,pb,sb,out,po,so,n) -> {
            var species=ByteVector.SPECIES_PREFERRED;
            long bound=species.loopBound(n);
            if(sa!=0 || so!=0 || sb!=1 || bound==0) {scalar.run(a,pa,sa,b,pb,sb,out,po,so,n);return;}
            boolean value=a.get(ValueLayout.JAVA_BYTE,pa)!=0;
            long i=0;
            for(;i<bound;i+=species.length()) {
                var mask=ByteVector.fromMemorySegment(species,b,pb+i,ByteOrder.nativeOrder()).compare(VectorOperators.NE,(byte)0);
                value=all?value && mask.allTrue():value || mask.anyTrue();
            }
            for(;i<n;i++) value=all?value && b.get(ValueLayout.JAVA_BYTE,pb+i)!=0:value || b.get(ValueLayout.JAVA_BYTE,pb+i)!=0;
            out.set(ValueLayout.JAVA_BYTE,po,(byte)(value?1:0));Accelerators.executed();
        };
    }
    static BinaryLoop extremaFLOAT(boolean maximum,BinaryLoop scalar) {
        return (a,pa,sa,b,pb,sb,out,po,so,n) -> {
            var species=FloatVector.SPECIES_PREFERRED;long bound=species.loopBound(n);
            if(sa!=0 || so!=0 || sb!=4 || bound==0) {scalar.run(a,pa,sa,b,pb,sb,out,po,so,n);return;}
            // Scalar ties pick an operand, while hardware min/max may choose a zero sign.
            // Detect those runs before any write and preserve the original scalar ordering.
            for(long i=0;i<bound;i+=species.length()) {
                var x=FloatVector.fromMemorySegment(species,b,pb+i*4,ByteOrder.nativeOrder());
                if(x.test(VectorOperators.IS_NAN).or(x.compare(VectorOperators.EQ,0)).anyTrue()) {
                    scalar.run(a,pa,sa,b,pb,sb,out,po,so,n);return;
                }
            }
            float value=a.get(ValueLayout.JAVA_FLOAT_UNALIGNED,pa);
            long i=0;
            for(;i<bound;i+=species.length()) {
                var x=FloatVector.fromMemorySegment(species,b,pb+i*4,ByteOrder.nativeOrder());
                float part=maximum?x.reduceLanes(VectorOperators.MAX):x.reduceLanes(VectorOperators.MIN);
                value=maximum?org.numjvm.internal.loops.ScalarMath.maximum(value,part):org.numjvm.internal.loops.ScalarMath.minimum(value,part);
            }
            for(;i<n;i++) {
                float part=b.get(ValueLayout.JAVA_FLOAT_UNALIGNED,pb+i*4);
                value=maximum?org.numjvm.internal.loops.ScalarMath.maximum(value,part):org.numjvm.internal.loops.ScalarMath.minimum(value,part);
            }
            out.set(ValueLayout.JAVA_FLOAT_UNALIGNED,po,value);Accelerators.executed();
        };
    }
    static BinaryLoop extremaDOUBLE(boolean maximum,BinaryLoop scalar) {
        return (a,pa,sa,b,pb,sb,out,po,so,n) -> {
            var species=DoubleVector.SPECIES_PREFERRED;long bound=species.loopBound(n);
            if(sa!=0 || so!=0 || sb!=8 || bound==0) {scalar.run(a,pa,sa,b,pb,sb,out,po,so,n);return;}
            // Scalar ties pick an operand, while hardware min/max may choose a zero sign.
            // Detect those runs before any write and preserve the original scalar ordering.
            for(long i=0;i<bound;i+=species.length()) {
                var x=DoubleVector.fromMemorySegment(species,b,pb+i*8,ByteOrder.nativeOrder());
                if(x.test(VectorOperators.IS_NAN).or(x.compare(VectorOperators.EQ,0)).anyTrue()) {
                    scalar.run(a,pa,sa,b,pb,sb,out,po,so,n);return;
                }
            }
            double value=a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,pa);
            long i=0;
            for(;i<bound;i+=species.length()) {
                var x=DoubleVector.fromMemorySegment(species,b,pb+i*8,ByteOrder.nativeOrder());
                double part=maximum?x.reduceLanes(VectorOperators.MAX):x.reduceLanes(VectorOperators.MIN);
                value=maximum?org.numjvm.internal.loops.ScalarMath.maximum(value,part):org.numjvm.internal.loops.ScalarMath.minimum(value,part);
            }
            for(;i<n;i++) {
                double part=b.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,pb+i*8);
                value=maximum?org.numjvm.internal.loops.ScalarMath.maximum(value,part):org.numjvm.internal.loops.ScalarMath.minimum(value,part);
            }
            out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,po,value);Accelerators.executed();
        };
    }
}
