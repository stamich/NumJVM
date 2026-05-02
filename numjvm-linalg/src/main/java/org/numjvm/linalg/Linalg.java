package org.numjvm.linalg;

import org.numjvm.ndarray.NDArray;

/**
 * Experimental float32/float64 linear algebra in the optional linalg artifact.
 * Matmul follows NumPy vector promotion and leading-dimension broadcasting;
 * dot additionally supports scalars and higher-rank contraction.
 */
@org.numjvm.Experimental
public final class Linalg {
    private Linalg() { }
    /**
     * Computes NumPy-style dot for float32 and float64 arrays.
     * Mixed types produce float64; the first scoped operand owns the result.
     * @param a left operand
     * @param b right operand
     * @return contraction result
     */
    public static NDArray dot(NDArray a, NDArray b) {
        MatmulMemory.accessible(a); MatmulMemory.accessible(b);
        var plan = ContractionPlan.dot(a.shape(), b.shape(), a.dtype(), b.dtype());
        var result = ContractionExecutor.allocate(a, b, plan);
        ContractionExecutor.dot(a, b, result, plan);
        return result;
    }
    /**
     * Multiplies two arrays. Mixed float32/64 produces float64. Results belong to
     * the first live operand scope, otherwise to automatic memory.
     * @param a left operand
     * @param b right operand
     * @return multiplication result
     */
    public static NDArray matmul(NDArray a, NDArray b) {
        return execute(a, b, null);
    }
    /**
     * Multiplies into exact-shape, exact-dtype writable output. Alias hazards use
     * independent storage and copyback. Unexpected execution failures may leave
     * partial output; updates are not transactional.
     * @param a left operand
     * @param b right operand
     * @param out caller-supplied output
     * @return out
     */
    public static NDArray matmul(NDArray a, NDArray b, NDArray out) {
        return execute(a, b, java.util.Objects.requireNonNull(out, "out"));
    }
    /** Returns available native provider/configuration or reason Java is used.
     * @return available native provider/configuration or reason Java is used
     */
    public static String nativeProvider() { return OpenBlas.availability(); }
    /** Returns last selection on the calling thread, separate from provider availability.
     * @return last selection on the calling thread, separate from provider availability
     */
    public static String lastBackend() { return BlasDispatch.last(); }
    /** Returns successfully returned native GEMM calls across the JVM.
     * @return successfully returned native GEMM calls across the JVM
     */
    public static long nativeCalls() { return BlasDispatch.calls(); }

    private static NDArray execute(NDArray a, NDArray b, NDArray out) {
        MatmulMemory.accessible(a); MatmulMemory.accessible(b);
        var plan = MatmulPlan.of(a.shape(), b.shape());
        var type = MatmulMemory.dtype(a, b);
        if (out != null) { MatmulMemory.output(out, plan, type); }
        BlasDispatch.initialize();
        boolean temporary = out != null && (!out.isCContiguous() || out.sharesMemoryWith(a) || out.sharesMemoryWith(b));
        NDArray result = out == null || temporary ? MatmulMemory.allocate(a, b, plan, type) : out;
        if (plan.output().size() == 0) { BlasDispatch.emptyBatch(); }
        BatchExecutor.execute(a, b, result, plan, BlasDispatch::run);
        if (temporary) {
            var indices = new org.numjvm.iteration.IndexIterator(plan.output());
            while (indices.hasNext()) {
                long[] index = indices.next();
                out.setDouble(index, result.getDouble(index));
            }
            return out;
        }
        return result;
    }
}
