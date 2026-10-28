package org.numjvm.dtype;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import org.junit.jupiter.api.Test;

class ComplexTest {

    @Test
    void toStringIsPythonRepr() {
        // expected values from Python 3: repr(complex(re, im))
        assertEquals("(1+2j)", Complex.of(1, 2).toString());
        assertEquals("(1.5-0.25j)", Complex.of(1.5, -0.25).toString());
        assertEquals("2j", Complex.of(0, 2).toString());
        assertEquals("(-0+2j)", Complex.of(-0.0, 2).toString());
        assertEquals("-0j", Complex.of(0, -0.0).toString());
        assertEquals("0j", Complex.ZERO.toString());
        assertEquals("(nan+infj)", Complex.of(Double.NaN, Double.POSITIVE_INFINITY).toString());
        assertEquals("(-inf-infj)", Complex.of(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY).toString());
        assertEquals("(1e+16+1e-05j)", Complex.of(1e16, 1e-5).toString());
        assertEquals("(1000000000000000+0.00015j)", Complex.of(1e15, 1.5e-4).toString());
        assertEquals("(-2.5e-300+1.2345678901234568e+17j)", Complex.of(-2.5e-300, 123456789012345678.0).toString());
        assertEquals("(0.1+0.3333333333333333j)", Complex.of(0.1, 1.0 / 3).toString());
        assertEquals("(1e+100+5e-324j)", Complex.of(1e100, Double.MIN_VALUE).toString());
    }

    @Test
    void factoriesAndConstants() {
        assertEquals(new Complex(2.5, 0), Complex.ofReal(2.5));
        assertEquals(new Complex(1, 0), Complex.ONE);
        assertEquals(new Complex(0, 1), Complex.I);
        assertEquals(3.0, Complex.of(3, 4).re());
        assertEquals(4.0, Complex.of(3, 4).im());
    }

    @Test
    void classification() {
        assertTrue(Complex.of(Double.NaN, 1).isNaN());
        assertTrue(Complex.of(1, Double.NaN).isNaN());
        assertFalse(Complex.of(1, Double.POSITIVE_INFINITY).isNaN());
        assertTrue(Complex.of(Double.NaN, Double.NEGATIVE_INFINITY).isInfinite());
        assertFalse(Complex.of(Double.NaN, 0).isInfinite());
        assertTrue(Complex.of(1, -2).isFinite());
        assertFalse(Complex.of(Double.NaN, 0).isFinite());
    }

    @Test
    void equalityIsBitwiseLikeDoubleCompare() {
        assertEquals(Complex.of(Double.NaN, 0), Complex.of(Double.NaN, 0));
        assertNotEquals(Complex.of(0.0, 0), Complex.of(-0.0, 0));
        assertEquals(Complex.of(1, 2).hashCode(), Complex.of(1, 2).hashCode());
    }

    @Property(tries = 2000)
    void partsReadBackFromTheString(@ForAll double re, @ForAll double im) {
        String s = Complex.of(re, im).toString();
        String body = s.startsWith("(") ? s.substring(1, s.length() - 2) : s.substring(0, s.length() - 1);
        int split = body.length();
        if (s.startsWith("(")) {
            // the sign between the parts is the last '+' or '-' not preceded by 'e'
            for (int i = body.length() - 1; i > 0; i--) {
                char c = body.charAt(i);
                if ((c == '+' || c == '-') && body.charAt(i - 1) != 'e') {
                    split = i;
                    break;
                }
            }
            assertEquals(Double.doubleToLongBits(re), Double.doubleToLongBits(parse(body.substring(0, split))), s);
        }
        assertEquals(Double.doubleToLongBits(im), Double.doubleToLongBits(parse(body.substring(
                s.startsWith("(") ? split : 0))), s);
    }

    private static double parse(String s) {
        String t = s.startsWith("+") ? s.substring(1) : s;
        return switch (t) {
            case "nan" -> Double.NaN;
            case "inf" -> Double.POSITIVE_INFINITY;
            case "-inf" -> Double.NEGATIVE_INFINITY;
            default -> Double.parseDouble(t);
        };
    }
}
