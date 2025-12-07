/**
 * Universal functions: element-wise operations with broadcasting and type promotion.
 *
 * <p>A ufunc ({@link org.numjvm.ufunc.UnaryUfunc} or {@link org.numjvm.ufunc.BinaryUfunc})
 * resolves its operands' dtypes to one inner loop (ADR-005: promote, then the first loop the
 * promoted dtype casts to safely), converts operands of another dtype, allocates the result
 * and runs the loop over an iteration plan (ADR-004). All ufuncs are constants of
 * {@link org.numjvm.ufunc.Ufuncs}; {@link org.numjvm.NJ} and {@link org.numjvm.ndarray.NDArray}
 * offer shortcuts for the common ones.
 *
 * <p>Results are new arrays in automatically managed memory, laid out in the memory order of
 * the operands (NumPy's {@code order='K'}): adding two transposed arrays gives a transposed,
 * contiguous result.
 */
package org.numjvm.ufunc;
