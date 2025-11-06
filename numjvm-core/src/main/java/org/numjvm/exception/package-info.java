/**
 * Exceptions thrown by NumJVM.
 *
 * <p>All are unchecked and share one root:
 * <pre>
 * RuntimeException
 * └── {@link org.numjvm.exception.NumJVMException NumJVMException}
 *     ├── {@link org.numjvm.exception.ShapeException ShapeException}         invalid shape, axis, stride, view, broadcast or reshape
 *     ├── {@link org.numjvm.exception.DTypeException DTypeException}         unknown or unsupported dtype, invalid conversion
 *     ├── {@link org.numjvm.exception.IndexException IndexException}         index outside an axis, wrong number of indices
 *     ├── {@link org.numjvm.exception.ReadOnlyArrayException ReadOnlyArrayException} write through a read-only view
 *     ├── {@link org.numjvm.exception.MemoryException MemoryException}       closed scope, wrong thread, allocation failure
 *     └── org.numjvm.io.NpyFormatException  malformed or unsupported .npy data (module org.numjvm.io)
 * </pre>
 *
 * <p>Catch {@code NumJVMException} to handle any NumJVM failure in one place.
 */
package org.numjvm.exception;
