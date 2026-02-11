package org.numjvm.linalg;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;

class MatmulPropertyTest {
    @Property(tries=40,seed="20261030")
    void rightIdentityPreservesFiniteMatrices(@ForAll @IntRange(min=0,max=6) int m,
                                             @ForAll @IntRange(min=0,max=7) int n,
                                             @ForAll boolean singlePrecision) {
        DType type=singlePrecision?DType.FLOAT32:DType.FLOAT64;
        var a=NJ.arange(0,m*n,1).astype(type).reshape(m,n);
        var identity=NJ.zeros(type,n,n);
        for(int i=0;i<n;i++)identity.setDouble(i,i,1);
        assertTrue(NJ.arrayEquals(a,Linalg.matmul(a,identity)));
    }
}
