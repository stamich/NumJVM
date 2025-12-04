package org.numjvm.internal.loops;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.math.BigInteger;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.dtype.Unsigned;
import org.numjvm.exception.DTypeException;
import org.numjvm.internal.HalfFloat;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Order;
import org.numjvm.ndarray.Slice;

class CastLoopsTest {

    @Test
    void everyPairOfDtypesHasACastExceptComplexToReal() {
        for (DType from : DType.values()) {
            for (DType to : DType.values()) {
                LoopSignature sig = KernelRegistry.find("cast:" + to.typeName(), from);
                if (from.isComplex() && !to.isComplex()) {
                    assertNull(sig, from + " -> " + to);
                } else {
                    assertNotNull(sig, from + " -> " + to);
                    assertEquals(to, sig.output());
                }
            }
        }
    }

    /**
     * Small integers 0..9 and their negatives where the dtype allows: exact in every dtype.
     */
    @ParameterizedTest
    @EnumSource(DType.class)
    void smallValuesConvertExactlyBetweenAllDtypes(DType from) {
        if (from.isComplex()) {
            return;
        }
        double[] v = from == DType.BOOL ? new double[]{0, 1, 1, 0} : from.isUnsigned()
                ? new double[]{0, 1, 7, 9} : new double[]{0, -1, 7, -9};
        NDArray src = NJ.array(from, v).reshape(2, 2).transpose();
        for (DType to : DType.values()) {
            NDArray dst = src.astype(to);
            NDArray back = to.isComplex() ? dst.astype(DType.COMPLEX128) : dst;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    double x = src.getDouble(i, j);
                    double expected = to == DType.BOOL ? (x != 0 ? 1 : 0)
                            : to.isUnsigned() && x < 0 ? -1 : x;
                    if (to.isComplex()) {
                        continue;
                    }
                    if (to.isUnsigned() && x < 0) {
                        // wraps: -1 -> max, -9 -> max - 8
                        long bits = back.getLong(i, j);
                        long width = to.size() * 8L;
                        long mask = width == 64 ? -1L : (1L << width) - 1;
                        assertEquals((long) x & mask, bits, from + " -> " + to);
                    } else {
                        assertEquals(expected, back.getDouble(i, j), from + " -> " + to);
                    }
                }
            }
        }
    }

    @Test
    void realToComplexHasZeroImaginaryPart() {
        NDArray c64 = NJ.array(new double[]{1.5, -2}).astype(DType.COMPLEX64);
        java.lang.foreign.MemorySegment s = org.numjvm.internal.SharedSecrets.arrayAccess().buffer(c64).segment();
        java.lang.foreign.ValueLayout.OfFloat f32 = java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
        assertEquals(1.5f, s.get(f32, 0));
        assertEquals(0f, s.get(f32, 4));
        assertEquals(-2f, s.get(f32, 8));
        assertEquals(0f, s.get(f32, 12));
        NDArray c128 = c64.astype(DType.COMPLEX128);
        java.lang.foreign.MemorySegment w = org.numjvm.internal.SharedSecrets.arrayAccess().buffer(c128).segment();
        java.lang.foreign.ValueLayout.OfDouble f64 = java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED;
        assertEquals(-2.0, w.get(f64, 16));
        assertEquals(0.0, w.get(f64, 24));
        NDArray back = c128.astype(DType.COMPLEX64);
        assertEquals(-2f, org.numjvm.internal.SharedSecrets.arrayAccess().buffer(back).segment().get(f32, 8));
        assertThrows(DTypeException.class, () -> c64.astype(DType.FLOAT64));
    }

    @Test
    void floatToIntegerTruncatesAndWraps() {
        NDArray f = NJ.array(new double[]{300.7, -1.5, 2.9, -128.9});
        NDArray i8 = f.astype(DType.INT8);
        assertEquals(44, i8.getLong(0));
        assertEquals(-1, i8.getLong(1));
        assertEquals(2, i8.getLong(2));
        assertEquals(-128, i8.getLong(3));
        assertEquals(-1L, NJ.array(new double[]{-1.0}).astype(DType.UINT64).getLong(0));
        assertEquals(0xFFFF_FFFFL, NJ.array(new double[]{-1.0}).astype(DType.UINT32).getLong(0));
    }

    @Test
    void float64ToFloat16RoundsOnce() {
        // 1 + 2^-11 + 2^-40 lies just above the midpoint between 1 and 1 + 2^-10.
        // NumPy: np.float16(1 + 2**-11 + 2**-40).view(np.uint16) == 0x3C01; via float32 it is 0x3C00.
        double d = 1 + 0x1p-11 + 0x1p-40;
        assertEquals((short) 0x3C01, HalfFloat.doubleToFloat16(d));
        assertEquals((short) 0x3C00, Float.floatToFloat16((float) d));
        NDArray h = NJ.array(new double[]{d}).astype(DType.FLOAT16);
        assertEquals(Float.float16ToFloat((short) 0x3C01), h.getDouble(0), 0.0);
        NDArray s = NJ.zeros(DType.FLOAT16, 1);
        s.setDouble(0, d);
        assertEquals(h.getDouble(0), s.getDouble(0), 0.0);
    }

    @Test
    void castsRunOnStridedAndBroadcastSources() {
        NDArray src = NJ.arange(0, 24, 1).reshape(4, 6).slice(Slice.step(-1), Slice.step(2));
        NDArray f = src.astype(DType.FLOAT32, Order.F);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(src.getLong(i, j), (long) f.getDouble(i, j));
            }
        }
        NDArray b = NJ.array(new double[]{1, 2, 3}).broadcastTo(4, 3).astype(DType.INT16);
        assertEquals(3, b.getLong(3, 2));
    }

    // ------------------------------------------------------------------ correctly rounded conversions

    @Property(tries = 3000)
    void int64ToFloat32IsCorrectlyRounded(@ForAll long v) {
        NDArray a = NJ.zeros(DType.INT64, 1);
        a.setLong(0, v);
        float f = (float) a.astype(DType.FLOAT32).getDouble(0);
        assertEquals(new java.math.BigDecimal(BigInteger.valueOf(v)).floatValue(), f);
    }

    @Property(tries = 3000)
    void uint64ToFloat32AndFloat64AreCorrectlyRounded(@ForAll long bits) {
        BigInteger exact = new BigInteger(Long.toUnsignedString(bits));
        assertEquals(new java.math.BigDecimal(exact).floatValue(), Unsigned.toFloat(bits));
        assertEquals(new java.math.BigDecimal(exact).doubleValue(), Unsigned.toDouble(bits));
        NDArray a = NJ.zeros(DType.UINT64, 1);
        a.setLong(0, bits);
        assertEquals(Unsigned.toFloat(bits), (float) a.astype(DType.FLOAT32).getDouble(0));
    }

    @Property(tries = 5000)
    void float64ToFloat16IsCorrectlyRounded(@ForAll long bits) {
        // map random bits into the float16 range so that most cases are finite
        double d = Double.longBitsToDouble(bits);
        if (!Double.isFinite(d)) {
            return;
        }
        double scaled = Math.scalb(d, -Math.getExponent(d) + (int) Math.floorMod(bits, 41L) - 25);
        assertEquals(nearestHalf(scaled), HalfFloat.doubleToFloat16(scaled), () -> Double.toString(scaled));
    }

    /**
     * Reference: the binary16 nearest to {@code d}, ties to even, by comparing with the exact
     * midpoints between neighbouring binary16 values (exact in double).
     */
    static short nearestHalf(double d) {
        short sign = (short) (Math.copySign(1.0, d) < 0 ? 0x8000 : 0);
        double a = Math.abs(d);
        if (a >= 65520.0) {
            return (short) (sign | 0x7C00);
        }
        int lo = 0;
        int hi = 0x7BFF;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (Float.float16ToFloat((short) mid) <= a) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        int h = lo;
        double v = Float.float16ToFloat((short) h);
        double next = Float.float16ToFloat((short) (h + 1));
        double midpoint = (v + next) / 2;
        int r;
        if (a < midpoint) {
            r = h;
        } else if (a > midpoint) {
            r = h + 1;
        } else {
            r = (h & 1) == 0 ? h : h + 1;
        }
        return (short) (sign | r);
    }

    @Property(tries = 3000)
    void float64ToBFloat16IsCorrectlyRounded(@ForAll long bits) {
        double d = Double.longBitsToDouble(bits);
        if (!Double.isFinite(d) || Math.abs(d) > 1e38 || Math.abs(d) < 1e-37) {
            return;
        }
        short got = HalfFloat.doubleToBFloat16(d);
        // neighbours: the result must be no farther from d than either adjacent bfloat16
        double v = HalfFloat.bfloat16ToFloat(got);
        double up = HalfFloat.bfloat16ToFloat((short) (got + 1));
        double down = HalfFloat.bfloat16ToFloat((short) (got - 1));
        double err = Math.abs(d - v);
        assertEquals(true, err <= Math.abs(d - up) && err <= Math.abs(d - down), () -> d + " -> " + v);
    }
}
