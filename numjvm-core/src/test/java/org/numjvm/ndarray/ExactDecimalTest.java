package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import org.junit.jupiter.api.Test;

class ExactDecimalTest {

    @Test
    void knownValues() {
        assertEquals("0.1000000000000000055511151231257827021181583404541015625",
                ArrayFormatter.exactDecimal(0.1).toPlainString());
        assertEquals("-2.5", ArrayFormatter.exactDecimal(-2.5).toPlainString());
        assertEquals("65504", ArrayFormatter.exactDecimal(65504.0).toPlainString());
        assertEquals(0, ArrayFormatter.exactDecimal(0.0).signum());
        assertEquals(0, ArrayFormatter.exactDecimal(-0.0).signum());
        // smallest float16 subnormal, 2^-24
        assertEquals("5.9604644775390625E-8", ArrayFormatter.exactDecimal(0x1p-24).toString());
        BigDecimal tiny = ArrayFormatter.exactDecimal(Double.MIN_VALUE);
        assertEquals(1074, tiny.scale());
        assertEquals(Double.MIN_VALUE, tiny.doubleValue());
    }

    @Test
    void rejectsNonFinite() {
        assertThrows(NumberFormatException.class, () -> ArrayFormatter.exactDecimal(Double.NaN));
        assertThrows(NumberFormatException.class, () -> ArrayFormatter.exactDecimal(Double.POSITIVE_INFINITY));
    }

    @Property
    void isExactAndMinimal(@ForAll long bits) {
        double v = Double.longBitsToDouble(bits);
        if (!Double.isFinite(v) || v == 0) {
            return;
        }
        BigDecimal d = ArrayFormatter.exactDecimal(v);
        assertEquals(v, d.doubleValue());
        // exact: d * 2^1074 is an integer for every finite double
        assertTrue(d.multiply(BigDecimal.TWO.pow(1074)).stripTrailingZeros().scale() <= 0);
        // minimal scale, as new BigDecimal(double) produces: no trailing zero digit after the point
        assertTrue(d.scale() <= 0 || d.unscaledValue().mod(java.math.BigInteger.TEN).signum() != 0, d::toString);
    }
}
