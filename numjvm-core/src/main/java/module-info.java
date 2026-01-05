/**
 * NumJVM core: multidimensional arrays for the JVM.
 *
 * <p>Entry point: {@link org.numjvm.NJ}.
 */
// "module" warnings are suppressed because of the qualified exports: org.numjvm.io is
// compiled after this module and is not on the module path when this one is built.
@SuppressWarnings("module")
module org.numjvm.core {
    exports org.numjvm;
    exports org.numjvm.dtype;
    exports org.numjvm.exception;
    exports org.numjvm.iteration;
    exports org.numjvm.memory;
    exports org.numjvm.ndarray;
    exports org.numjvm.shape;
    exports org.numjvm.ufunc;

    exports org.numjvm.internal to org.numjvm.io;
    exports org.numjvm.internal.memory to org.numjvm.io;
}
