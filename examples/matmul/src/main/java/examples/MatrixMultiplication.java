package examples;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.linalg.Linalg;

/** Java-only by default; force OpenBLAS with the documented Gradle properties. */
public final class MatrixMultiplication {
    private MatrixMultiplication() { }
    public static void main(String[] args) {
        try(var scope=NJ.scope()) {
            var a=scope.ones(DType.FLOAT64,2,3);
            var b=NJ.ones(3,4);
            var c=Linalg.matmul(a,b);
            if(c.getDouble(1,3)!=3 || c.memoryOwner()!=scope)throw new AssertionError("value or scope mismatch");
            var out=NJ.zeros(2,4);
            Linalg.matmul(a,b,out);
            if(out.getDouble(0,0)!=3)throw new AssertionError("out failed");
            System.out.println(c);
            System.out.println("provider="+Linalg.nativeProvider()+", execution="+Linalg.lastBackend()
                    +", native calls="+Linalg.nativeCalls());
        }
    }
}
