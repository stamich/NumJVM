package org.numjvm.vector;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.*;
import org.numjvm.internal.backend.Accelerators;
import org.numjvm.internal.loops.BinaryLoop;
import org.numjvm.dtype.DType;
/** Eight logical accumulators and recursive 128-element blocks, independent of lane width. */
final class VectorPairwise {
    private VectorPairwise() { }
    static BinaryLoop wrap(DType type, BinaryLoop scalar, BinaryLoop elementwise) {
        return (a,pa,sa,b,pb,sb,out,po,so,n) -> {
            if(sa==0 && so==0 && a==out && pa==po && sb==type.size() && n>=8) {
                if(type==DType.FLOAT64) out.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,po,
                    a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,pa)+float64(b,pb,n));
                else out.set(ValueLayout.JAVA_FLOAT_UNALIGNED,po,
                    a.get(ValueLayout.JAVA_FLOAT_UNALIGNED,pa)+float32(b,pb,n));
            } else elementwise.run(a,pa,sa,b,pb,sb,out,po,so,n);
        };
    }
    static double float64(MemorySegment s,long off,long n) {
        if(n<8) return org.numjvm.internal.loops.Pairwise.float64(s,off,8,n);
        if(n>128) { long h=n/2;h-=h%8;return float64(s,off,h)+float64(s,off+h*8,n-h); }
        int bits=Integer.getInteger("numjvm.vector.bits",DoubleVector.SPECIES_PREFERRED.vectorBitSize());
        var species=VectorSpecies.of(double.class,VectorShape.forBitSize(Math.min(bits,512)));
        int lanes=species.length();
        var accum=new DoubleVector[8/lanes];
        for(int g=0;g<accum.length;g++) accum[g]=DoubleVector.fromMemorySegment(species,s,off+g*lanes*8,ByteOrder.nativeOrder());
        long limit=n-n%8;
        for(long i=8;i<limit;i+=8) {
            for(int g=0;g<accum.length;g++) accum[g]=accum[g].add(
                DoubleVector.fromMemorySegment(species,s,off+(i+g*lanes)*8,ByteOrder.nativeOrder()));
        }
        double[] r=new double[8];
        for(int g=0;g<accum.length;g++) accum[g].intoArray(r,g*lanes);
        double result=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(long i=limit;i<n;i++) result+=s.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,off+i*8);
        Accelerators.executed();
        return result;
    }
    static float float32(MemorySegment s,long off,long n) {
        if(n<8) return org.numjvm.internal.loops.Pairwise.float32(s,off,4,n);
        if(n>128) { long h=n/2;h-=h%8;return float32(s,off,h)+float32(s,off+h*4,n-h); }
        int bits=Integer.getInteger("numjvm.vector.bits",FloatVector.SPECIES_PREFERRED.vectorBitSize());
        var species=VectorSpecies.of(float.class,VectorShape.forBitSize(Math.min(bits,256)));
        int lanes=species.length();
        var accum=new FloatVector[8/lanes];
        for(int g=0;g<accum.length;g++) accum[g]=FloatVector.fromMemorySegment(species,s,off+g*lanes*4,ByteOrder.nativeOrder());
        long limit=n-n%8;
        for(long i=8;i<limit;i+=8) {
            for(int g=0;g<accum.length;g++) accum[g]=accum[g].add(
                FloatVector.fromMemorySegment(species,s,off+(i+g*lanes)*4,ByteOrder.nativeOrder()));
        }
        float[] r=new float[8];
        for(int g=0;g<accum.length;g++) accum[g].intoArray(r,g*lanes);
        float result=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(long i=limit;i<n;i++) result+=s.get(ValueLayout.JAVA_FLOAT_UNALIGNED,off+i*4);
        Accelerators.executed();
        return result;
    }
}
