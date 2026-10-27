package org.numjvm.io;

/**
 * Limits applied when reading {@code .npy} data, so that a malformed or malicious file
 * cannot make the reader allocate unbounded memory.
 *
 * @param maxHeaderBytes largest header (the Python dict literal) accepted, in bytes
 * @param maxRank        largest number of dimensions accepted
 * @param maxDataBytes   largest array payload accepted, in bytes
 */
public record NpyLimits(int maxHeaderBytes, int maxRank, long maxDataBytes) {

    /**
     * NumPy's own default for {@code np.load(max_header_size=...)}.
     */
    public static final int DEFAULT_MAX_HEADER_BYTES = 10_000;

    /**
     * Validates the limits.
     *
     * @param maxHeaderBytes largest header, {@code > 0}
     * @param maxRank        largest rank, {@code 0..64}
     * @param maxDataBytes   largest payload, {@code >= 0}
     */
    public NpyLimits {
        if (maxHeaderBytes <= 0 || maxRank < 0 || maxRank > 64 || maxDataBytes < 0) {
            throw new IllegalArgumentException("invalid limits: header " + maxHeaderBytes + ", rank " + maxRank
                    + ", data " + maxDataBytes);
        }
    }

    /**
     * The defaults: 10,000-byte headers (as NumPy), rank 64, no limit on payload size
     * beyond what the file actually contains.
     *
     * @return the default limits
     */
    public static NpyLimits defaults() {
        return new NpyLimits(DEFAULT_MAX_HEADER_BYTES, 64, Long.MAX_VALUE);
    }

    /**
     * Returns a copy with a different payload limit.
     *
     * @param bytes largest payload accepted
     * @return a copy with a different payload limit
     */
    public NpyLimits withMaxDataBytes(long bytes) {
        return new NpyLimits(maxHeaderBytes, maxRank, bytes);
    }

    /**
     * Returns a copy with a different header limit.
     *
     * @param bytes largest header accepted
     * @return a copy with a different header limit
     */
    public NpyLimits withMaxHeaderBytes(int bytes) {
        return new NpyLimits(bytes, maxRank, maxDataBytes);
    }
}
