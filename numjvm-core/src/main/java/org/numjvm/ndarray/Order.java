package org.numjvm.ndarray;

/**
 * Memory order of a contiguous array.
 */
public enum Order {

    /**
     * Row-major: the last axis varies fastest. The default everywhere in NumJVM.
     */
    C,

    /**
     * Column-major (Fortran): the first axis varies fastest.
     */
    F
}
