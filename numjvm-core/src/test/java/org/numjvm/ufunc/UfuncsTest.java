package org.numjvm.ufunc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.ndarray.NDArray;

class UfuncsTest {

    @Test
    void everyConstantIsListedAndFoundByName() throws IllegalAccessException {
        Set<Ufunc> constants = new HashSet<>();
        for (Field f : Ufuncs.class.getFields()) {
            if (Modifier.isStatic(f.getModifiers()) && f.get(null) instanceof Ufunc u) {
                constants.add(u);
                assertSame(u, Ufuncs.byName(u.name()).orElseThrow(), f.getName());
                assertEquals(f.getName(), u.name().toUpperCase(java.util.Locale.ROOT), "constant name");
            }
        }
        assertEquals(constants, new HashSet<>(Ufuncs.all()));
        assertTrue(Ufuncs.byName("no_such_ufunc").isEmpty());
    }

    @Test
    void subtractAndMultiply() {
        NDArray a = NJ.array(new long[]{10, -3, 7});
        NDArray b = NJ.array(new long[]{4, 5, -2});
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{6, -8, 9}), NJ.subtract(a, b)));
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{40, -15, -14}), a.multiply(b)));
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{9, -4, 6}), a.subtract(1L)));
        assertEquals(DType.FLOAT64, NJ.multiply(a, 0.5).dtype());
        assertEquals(-1.5, a.multiply(0.5).getDouble(1));
    }

    @Test
    void boolSubtractIsRejectedAndMultiplyIsAnd() {
        NDArray t = NJ.array(new boolean[]{true, true, false});
        NDArray f = NJ.array(new boolean[]{true, false, false});
        DTypeException e = assertThrows(DTypeException.class, () -> NJ.subtract(t, f));
        assertTrue(e.getMessage().contains("boolean subtract"), e.getMessage());
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, false, false}), NJ.multiply(t, f)));
        // bool with a non-bool operand promotes and subtracts normally
        assertEquals(DType.INT8, NJ.subtract(t, NJ.array(DType.INT8, new double[]{1, 1, 1})).dtype());
    }
}
