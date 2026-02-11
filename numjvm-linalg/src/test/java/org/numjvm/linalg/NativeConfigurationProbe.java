package org.numjvm.linalg;
public final class NativeConfigurationProbe {
    private NativeConfigurationProbe() { }
    public static void main(String[] args) {
        var result=Linalg.matmul(org.numjvm.NJ.ones(2,2),org.numjvm.NJ.ones(2,2));
        if(result.getDouble(0,0)!=2)throw new AssertionError("fallback wrong");
        System.out.println(Linalg.lastBackend());
    }
}
