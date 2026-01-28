package org.numjvm.linalg;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.numjvm.dtype.DType;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.ndarray.NDArray;

/** Logical matrix addressing; negative/zero strides are valid inputs. */
record MatrixView(MemorySegment memory,long base,long rows,long columns,DType type) {
    static MatrixView input(NDArray a,boolean left) {
        var access=SharedSecrets.arrayAccess();
        long r=a.ndim()==1?(left?0:a.strides().stride(0)):a.strides().stride(a.ndim()-2);
        long c=a.ndim()==1?(left?a.strides().stride(0):0):a.strides().stride(a.ndim()-1);
        return new MatrixView(access.buffer(a).segment(),access.offset(a),r,c,a.dtype());
    }
    static MatrixView output(NDArray a,MatmulPlan p) {
        var access=SharedSecrets.arrayAccess();int rank=a.ndim();
        long r=p.leftVector()?0:a.strides().stride(rank-(p.rightVector()?1:2));
        long c=p.rightVector()?0:a.strides().stride(rank-1);
        return new MatrixView(access.buffer(a).segment(),access.offset(a),r,c,a.dtype());
    }
    double get(long i,long j) {
        long offset=base+i*rows+j*columns;
        return type==DType.FLOAT64?memory.get(ValueLayout.JAVA_DOUBLE_UNALIGNED,offset):memory.get(ValueLayout.JAVA_FLOAT_UNALIGNED,offset);
    }
    void set(long i,long j,double value) {
        long offset=base+i*rows+j*columns;
        if(type==DType.FLOAT64)memory.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,offset,value);
        else memory.set(ValueLayout.JAVA_FLOAT_UNALIGNED,offset,(float)value);
    }
    MatrixView at(long offset) { return new MatrixView(memory,base+offset,rows,columns,type); }
}
