package org.numjvm.internal.loops;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigInteger;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.constraints.IntRange;
import net.jqwik.api.constraints.LongRange;
import org.junit.jupiter.api.Test;
import org.numjvm.exception.ValueException;

/**
 * Expected values are NumPy 2.5's.
 */
class ScalarMathTest {

    private static void assertBits(double expected, double actual) {
        assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual),
                () -> "expected " + expected + " but was " + actual);
    }

    @Test
    void integerDivisionByZeroIsZero() {
        // np.floor_divide(np.array([5, -5, 0], np.int32), 0) == [0, 0, 0]; remainder likewise
        for (int x : new int[]{5, -5, 0, Integer.MIN_VALUE}) {
            assertEquals(0, ScalarMath.floorDivide(x, 0));
            assertEquals(0, ScalarMath.remainder(x, 0));
            assertEquals(0L, ScalarMath.floorDivide((long) x, 0L));
            assertEquals(0L, ScalarMath.remainder((long) x, 0L));
        }
        assertEquals(0L, ScalarMath.floorDivideUnsigned(-1L, 0L));
        assertEquals(0L, ScalarMath.remainderUnsigned(-1L, 0L));
    }

    @Test
    void integerFloorDivisionRoundsDown() {
        assertEquals(-3, ScalarMath.floorDivide(-7, 3));
        assertEquals(2, ScalarMath.remainder(-7, 3));
        assertEquals(-3, ScalarMath.floorDivide(7, -3));
        assertEquals(-2, ScalarMath.remainder(7, -3));
        // np.int8(-128) // np.int8(-1) == -128: computed in int, wrapped when written as int8
        assertEquals(-128, (byte) ScalarMath.floorDivide(-128, -1));
        assertEquals(Integer.MIN_VALUE, ScalarMath.floorDivide(Integer.MIN_VALUE, -1));
        assertEquals(Long.MIN_VALUE, ScalarMath.floorDivide(Long.MIN_VALUE, -1L));
        // uint64: np.floor_divide(2**64 - 1, 2) == 2**63 - 1, np.remainder(2**64 - 1, 10) == 5
        assertEquals(Long.MAX_VALUE, ScalarMath.floorDivideUnsigned(-1L, 2L));
        assertEquals(5L, ScalarMath.remainderUnsigned(-1L, 10L));
    }

    @Property
    void integerDivisionIdentity(@ForAll int x, @ForAll int y) {
        if (y == 0) {
            return;
        }
        int q = ScalarMath.floorDivide(x, y);
        int r = ScalarMath.remainder(x, y);
        assertEquals(x, q * y + r);
        assertTrue(r == 0 || (r < 0) == (y < 0));
    }

    @Test
    void floatDivisionByZero() {
        // np.floor_divide([5., -5., 0., 1.], [0., 0., 0., -0.]) == [inf, -inf, nan, -inf]
        assertBits(Double.POSITIVE_INFINITY, ScalarMath.floorDivide(5.0, 0.0));
        assertBits(Double.NEGATIVE_INFINITY, ScalarMath.floorDivide(-5.0, 0.0));
        assertTrue(Double.isNaN(ScalarMath.floorDivide(0.0, 0.0)));
        assertBits(Double.NEGATIVE_INFINITY, ScalarMath.floorDivide(1.0, -0.0));
        // np.remainder([5., -5., 0.], 0.) == [nan, nan, nan]
        assertTrue(Double.isNaN(ScalarMath.remainder(5.0, 0.0)));
        assertTrue(Double.isNaN(ScalarMath.remainder(0.0, 0.0)));
        assertTrue(Float.isNaN(ScalarMath.remainder(5.0f, 0.0f)));
        assertEquals(Float.NEGATIVE_INFINITY, ScalarMath.floorDivide(-5.0f, 0.0f));
    }

    @Test
    void floatRemainderHasTheSignOfTheDivisor() {
        // np.remainder([-7, 7, -7.5, 7.5], [3, -3, 2, -2]) == [2, -2, 0.5, -0.5]
        assertBits(2.0, ScalarMath.remainder(-7.0, 3.0));
        assertBits(-2.0, ScalarMath.remainder(7.0, -3.0));
        assertBits(0.5, ScalarMath.remainder(-7.5, 2.0));
        assertBits(-0.5, ScalarMath.remainder(7.5, -2.0));
        assertBits(-4.0, ScalarMath.floorDivide(-7.5, 2.0));
        // zero results take the divisor's sign: np.remainder(6., -3.) == -0.0
        assertBits(-0.0, ScalarMath.remainder(6.0, -3.0));
        assertBits(0.0, ScalarMath.remainder(-6.0, 3.0));
        // zero quotients take the sign of a / b: np.floor_divide(0., -1.) == -0.0
        assertBits(-0.0, ScalarMath.floorDivide(0.0, -1.0));
        assertEquals(-0.0f, ScalarMath.floorDivide(0.0f, -1.0f));
        assertEquals(0.5f, ScalarMath.remainder(-7.5f, 2.0f));
        assertEquals(-4.0f, ScalarMath.floorDivide(-7.5f, 2.0f));
        assertTrue(Double.isNaN(ScalarMath.remainder(Double.NaN, 2.0)));
        assertTrue(Double.isNaN(ScalarMath.floorDivide(Double.NaN, 2.0)));
        assertTrue(Double.isNaN(ScalarMath.remainder(Double.POSITIVE_INFINITY, 2.0)));
    }

    @Property
    void floatDivmodIdentity(@ForAll @LongRange(min = -100000, max = 100000) long ia,
                             @ForAll @LongRange(min = -1000, max = 1000) long ib) {
        if (ib == 0) {
            return;
        }
        double a = ia / 8.0;
        double b = ib / 4.0;
        double q = ScalarMath.floorDivide(a, b);
        double r = ScalarMath.remainder(a, b);
        assertEquals(a, q * b + r, 1e-9);
        assertEquals(Math.floor(a / b), q, 0.0);
    }

    @Test
    void maximumAndMinimumPropagateNaN() {
        assertTrue(Double.isNaN(ScalarMath.maximum(Double.NaN, 1.0)));
        assertTrue(Double.isNaN(ScalarMath.maximum(1.0, Double.NaN)));
        assertTrue(Double.isNaN(ScalarMath.minimum(Double.NaN, 1.0)));
        assertTrue(Double.isNaN(ScalarMath.minimum(1.0, Double.NaN)));
        assertEquals(1.0, ScalarMath.fmax(Double.NaN, 1.0));
        assertEquals(1.0, ScalarMath.fmax(1.0, Double.NaN));
        assertEquals(1.0, ScalarMath.fmin(Double.NaN, 1.0));
        assertTrue(Double.isNaN(ScalarMath.fmax(Double.NaN, Double.NaN)));
        assertTrue(Float.isNaN(ScalarMath.maximum(1f, Float.NaN)));
        assertEquals(2f, ScalarMath.fmin(Float.NaN, 2f));
        assertEquals(3.0, ScalarMath.maximum(3.0, 2.0));
        assertEquals(2.0, ScalarMath.minimum(3.0, 2.0));
    }

    @Test
    void signedZeroTiesFollowNumpy() {
        // np.maximum(-0.0, 0.0) == 0.0 and np.maximum(0.0, -0.0) == -0.0: the second operand
        assertBits(0.0, ScalarMath.maximum(-0.0, 0.0));
        assertBits(-0.0, ScalarMath.maximum(0.0, -0.0));
        assertBits(0.0, ScalarMath.minimum(-0.0, 0.0));
        assertBits(-0.0, ScalarMath.minimum(0.0, -0.0));
        // np.fmax(-0.0, 0.0) == -0.0: the first operand
        assertBits(-0.0, ScalarMath.fmax(-0.0, 0.0));
        assertBits(0.0, ScalarMath.fmax(0.0, -0.0));
        assertBits(-0.0, ScalarMath.fmin(-0.0, 0.0));
    }

    @Test
    void integerPowerWraps() {
        // np.power(np.array([2, 3], np.int8), np.array([7, 5], np.int8)) == [-128, -13]
        assertEquals(-128, (byte) ScalarMath.power(2, 7));
        assertEquals(-13, (byte) ScalarMath.power(3, 5));
        assertEquals(1, ScalarMath.power(0, 0));
        assertEquals(0, ScalarMath.power(0, 3));
        assertEquals(1, ScalarMath.power(-1, 0));
        assertEquals(-1, ScalarMath.power(-1, 3));
        // np.power(np.uint64(3), np.uint64(40)) == 12157665459056928801
        assertEquals(new BigInteger("12157665459056928801").longValue(), ScalarMath.powerUnsigned(3, 40));
        assertEquals(1L << 62, ScalarMath.power(2L, 62L));
        ValueException e = assertThrows(ValueException.class, () -> ScalarMath.power(2, -1));
        assertEquals("Integers to negative integer powers are not allowed.", e.getMessage());
        assertThrows(ValueException.class, () -> ScalarMath.power(2L, -1L));
    }

    @Property
    void integerPowerMatchesBigInteger(@ForAll long x, @ForAll @IntRange(min = 0, max = 200) int y) {
        assertEquals(BigInteger.valueOf(x).pow(y).longValue(), ScalarMath.power(x, (long) y));
        assertEquals(BigInteger.valueOf(x).pow(y).intValue(), ScalarMath.power((int) x, y));
    }
}
