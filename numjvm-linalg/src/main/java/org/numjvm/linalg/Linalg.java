package org.numjvm.linalg;

import org.numjvm.ndarray.NDArray;

/**
 * Experimental float32/float64 matrix multiplication, following NumPy matmul
 * vector promotion and leading-dimension broadcasting. Rank-zero inputs and
 * integer/complex dtypes are not supported. Use the optional linalg artifact.
 */
public final class Linalg {
    private Linalg() { }
    /**
     * Multiplies two arrays. Mixed float32/64 produces float64. Results belong to
     * the first live operand scope, otherwise to automatic memory.
     * @param a left operand
     * @param b right operand
     * @return multiplication result
     */
    public static NDArray matmul(NDArray a, NDArray b) {
        throw new UnsupportedOperationException("matrix executor is not integrated yet");
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
        throw new UnsupportedOperationException("matrix executor is not integrated yet");
    }
}
