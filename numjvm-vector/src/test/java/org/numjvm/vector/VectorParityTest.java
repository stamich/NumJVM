package org.numjvm.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;
import org.numjvm.BackendInfo;
import org.numjvm.dtype.DType;
import org.numjvm.internal.loops.KernelRegistry;
import static org.junit.jupiter.api.Assertions.*;

class VectorParityTest {
    private static final double[] VALUES={0.0,-0.0,Double.NaN,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,
            Double.MIN_VALUE,-Double.MIN_VALUE,1e30,-1e30,1,-1,3.25,-9};
    private static void fill(MemorySegment m,DType t,int n,int phase) {
        for(int i=0;i<n;i++) {
            long p=8L+(long)i*t.size(); double v=VALUES[(i+phase)%VALUES.length];
            switch(t) {
                case FLOAT64 -> m.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,p,v);
                case FLOAT32 -> m.set(ValueLayout.JAVA_FLOAT_UNALIGNED,p,(float)v);
                case INT32 -> m.set(ValueLayout.JAVA_INT_UNALIGNED,p,i%2==0?Integer.MIN_VALUE+i:Integer.MAX_VALUE-i);
                case INT64 -> m.set(ValueLayout.JAVA_LONG_UNALIGNED,p,i%2==0?Long.MIN_VALUE+i:Long.MAX_VALUE-i);
                default -> throw new AssertionError(t);
            }
        }
    }
    private static void same(MemorySegment a,MemorySegment b,DType t,int n) {
        for(int i=0;i<n;i++) {
            long p=8L+(long)i*t.size();
            if(t==DType.FLOAT64 && Double.isNaN(a.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,p))) {
                assertTrue(Double.isNaN(b.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,p)));continue;
            }
            if(t==DType.FLOAT32 && Float.isNaN(a.get(ValueLayout.JAVA_FLOAT_UNALIGNED,p))) {
                assertTrue(Float.isNaN(b.get(ValueLayout.JAVA_FLOAT_UNALIGNED,p)));continue;
            }
            for(int j=0;j<t.size();j++) assertEquals(a.get(ValueLayout.JAVA_BYTE,p+j),b.get(ValueLayout.JAVA_BYTE,p+j),t+" at "+i);
        }
        assertEquals(0,b.get(ValueLayout.JAVA_LONG_UNALIGNED,0));
        assertEquals(0,b.get(ValueLayout.JAVA_BYTE,8L+(long)n*t.size()));
    }
    @Test void everyGeneratedBinaryMatchesScalarOnHeapNativeOffsetsBroadcastAndTails() {
        long before=BackendInfo.vectorRuns();
        for(boolean nativeMemory:new boolean[]{false,true}) try(Arena arena=Arena.ofConfined()) {
            for(DType t:new DType[]{DType.FLOAT32,DType.FLOAT64,DType.INT32,DType.INT64}) {
                String[] ops=t.isInteger()?new String[]{"add","subtract","multiply","bitwise_and","bitwise_or","bitwise_xor","equal","not_equal","less","less_equal","greater","greater_equal"}
                    :new String[]{"add","subtract","multiply","divide","maximum","minimum","fmax","fmin","equal","not_equal","less","less_equal","greater","greater_equal"};
                for(String op:ops) for(int n:new int[]{0,1,7,8,9,17,127,128,129,257}) for(int mode=0;mode<4;mode++) {
                    var a=nativeMemory?arena.allocate(4096,64):MemorySegment.ofArray(new long[512]);
                    var b=nativeMemory?arena.allocate(4096,64):MemorySegment.ofArray(new long[512]);
                    var expected=MemorySegment.ofArray(new long[512]);var actual=MemorySegment.ofArray(new long[512]);
                    fill(a,t,n,0);fill(b,t,n,3);
                    var sig=KernelRegistry.find(op,t,t);DType out=sig.output();
                    long sa=mode==1?0:mode==3?-t.size():t.size(),sb=mode==2?0:t.size();
                    long pa=mode==3 && n>0?8L+(n-1L)*t.size():8;
                    var scalar=KernelRegistry.scalarBinary(op,t,t);
                    scalar.run(a,pa,sa,b,8,sb,expected,8,out.size(),n);
                    new VectorProvider().binary(op,t,scalar).run(a,pa,sa,b,8,sb,actual,8,out.size(),n);
                    same(expected,actual,out,n);
                }
            }
        }
        assertTrue(BackendInfo.vectorRuns()>before);
    }
    @Test void unaryFiniteBitsAndExceptionalClassificationMatch() {
        for(DType t:new DType[]{DType.FLOAT32,DType.FLOAT64}) for(String op:new String[]{"negative","absolute","square","sqrt"}) {
            var in=MemorySegment.ofArray(new long[512]);fill(in,t,257,0);
            var expected=MemorySegment.ofArray(new long[512]);var actual=MemorySegment.ofArray(new long[512]);
            var scalar=KernelRegistry.scalarUnary(op,t);
            scalar.run(in,8,t.size(),expected,8,t.size(),257);
            new VectorProvider().unary(op,t,scalar).run(in,8,t.size(),actual,8,t.size(),257);
            same(expected,actual,t,257);
        }
    }
    @Test void pairwiseTreeIsIndependentOfSpeciesWidth() {
        String old=System.getProperty("numjvm.vector.bits");
        try {
            for(int bits:new int[]{64,128,256,512}) {
                System.setProperty("numjvm.vector.bits",""+bits);
                for(DType t:new DType[]{DType.FLOAT32,DType.FLOAT64}) for(int n:new int[]{7,8,9,127,128,129,257,513}) {
                    var in=MemorySegment.ofArray(new long[1024]);
                    for(int i=0;i<n;i++) {
                        double v=i%3==0?1e20:i%3==1?-1e20:1;
                        if(t==DType.FLOAT64) in.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,i*8L,v);
                        else in.set(ValueLayout.JAVA_FLOAT_UNALIGNED,i*4L,(float)v);
                    }
                    if(t==DType.FLOAT64) assertEquals(Double.doubleToRawLongBits(org.numjvm.internal.loops.Pairwise.float64(in,0,8,n)),Double.doubleToRawLongBits(VectorPairwise.float64(in,0,n)));
                    else assertEquals(Float.floatToRawIntBits(org.numjvm.internal.loops.Pairwise.float32(in,0,4,n)),Float.floatToRawIntBits(VectorPairwise.float32(in,0,n)));
                }
            }
        } finally { if(old==null)System.clearProperty("numjvm.vector.bits");else System.setProperty("numjvm.vector.bits",old); }
    }
}
