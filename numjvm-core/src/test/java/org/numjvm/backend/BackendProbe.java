package org.numjvm.backend;
/** Separate JVM entrypoint: backend initialization is intentionally immutable. */
public final class BackendProbe {
    private BackendProbe() { }
    public static void main(String[] args) { System.out.println(org.numjvm.BackendInfo.simdProvider()); }
}
