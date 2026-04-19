/** Experimental expression compilation; the public API is under development. */
module org.numjvm.jit {
    requires transitive org.numjvm.core;
    uses org.numjvm.internal.backend.FusedProvider;
    exports org.numjvm.jit;
}
