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

    @Test
    void divisionResultDtypes() {
        assertEquals(DType.FLOAT64, Ufuncs.DIVIDE.resultType(DType.INT8, DType.INT8));
        assertEquals(DType.FLOAT64, Ufuncs.DIVIDE.resultType(DType.BOOL, DType.BOOL));
        assertEquals(DType.FLOAT16, Ufuncs.DIVIDE.resultType(DType.FLOAT16, DType.INT8));
        assertEquals(DType.INT8, Ufuncs.FLOOR_DIVIDE.resultType(DType.BOOL, DType.BOOL));
        assertEquals(DType.INT8, Ufuncs.REMAINDER.resultType(DType.BOOL, DType.BOOL));
        assertEquals(DType.UINT64, Ufuncs.FLOOR_DIVIDE.resultType(DType.UINT64, DType.UINT8));
        assertEquals(DType.BFLOAT16, Ufuncs.DIVIDE.resultType(DType.BFLOAT16, DType.BFLOAT16));
    }

    @Test
    void divisionValues() {
        NDArray a = NJ.array(new long[]{7, -7, 7, 1});
        NDArray b = NJ.array(new long[]{2, 2, 0, 0});
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{3, -4, 0, 0}), a.floorDivide(b)));
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{1, 1, 0, 0}), a.remainder(b)));
        NDArray q = NJ.divide(a, b);
        assertEquals(3.5, q.getDouble(0));
        assertEquals(Double.POSITIVE_INFINITY, q.getDouble(2));
        assertEquals(-1, NJ.remainder(NJ.array(new double[]{5}), -3L).getDouble(0));
        assertEquals(DType.FLOAT64, NJ.floorDivide(a, 2.0).dtype());
    }

    @Test
    void powerAndExtrema() {
        assertEquals(DType.INT8, Ufuncs.POWER.resultType(DType.BOOL, DType.BOOL));
        assertTrue(NJ.arrayEquals(NJ.array(new long[]{1, 8, 81}), NJ.power(NJ.array(new long[]{1, 2, 3}),
                NJ.array(new long[]{5, 3, 4}))));
        assertThrows(org.numjvm.exception.ValueException.class,
                () -> NJ.power(NJ.array(new long[]{2}), NJ.array(new long[]{-1})));
        NDArray x = NJ.array(new double[]{Double.NaN, 1, 5});
        NDArray y = NJ.array(new double[]{2, Double.NaN, 3});
        assertTrue(Double.isNaN(NJ.maximum(x, y).getDouble(0)));
        assertTrue(Double.isNaN(x.minimum(y).getDouble(1)));
        assertEquals(2.0, NJ.fmax(x, y).getDouble(0));
        assertEquals(1.0, NJ.fmin(x, y).getDouble(1));
        assertEquals(5.0, x.maximum(y).getDouble(2));
        NDArray u = NJ.zeros(DType.UINT64, 2);
        u.setLong(0, -1L);  // 2^64 - 1
        u.setLong(1, 1L);
        assertEquals(-1L, NJ.maximum(u, 7L).getLong(0));
        assertEquals(1L, NJ.minimum(u, 7L).getLong(1));
    }

    @Test
    void signedZeroTiesFollowNumpyPerDtype() {
        // NumPy 2.5: float32/float64 maximum and minimum keep y on a tie, float16 keeps x
        for (DType t : new DType[]{DType.FLOAT32, DType.FLOAT64}) {
            NDArray pz = NJ.array(t, new double[]{0.0});
            NDArray nz = NJ.array(t, new double[]{-0.0});
            assertEquals(-0.0, NJ.maximum(pz, nz).getDouble(0), t.typeName());
            assertEquals(0.0, NJ.minimum(nz, pz).getDouble(0), t.typeName());
        }
        NDArray pz = NJ.array(DType.FLOAT16, new double[]{0.0});
        NDArray nz = NJ.array(DType.FLOAT16, new double[]{-0.0});
        assertEquals(0.0, NJ.maximum(pz, nz).getDouble(0));
        assertEquals(-0.0, NJ.maximum(nz, pz).getDouble(0));
        assertEquals(-0.0, NJ.minimum(nz, pz).getDouble(0));
        assertEquals(0.0, NJ.minimum(pz, nz).getDouble(0));
        assertTrue(Double.isNaN(NJ.maximum(pz, NJ.array(DType.FLOAT16, new double[]{Double.NaN})).getDouble(0)));
    }

    @Test
    void comparisons() {
        NDArray x = NJ.array(new double[]{1, Double.NaN, 3});
        NDArray y = NJ.array(new double[]{1, Double.NaN, 2});
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, false, false}), NJ.equal(x, y)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, true, true}), x.notEqual(y)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, false, true}), x.greater(y)));
        assertEquals(DType.BOOL, Ufuncs.LESS.resultType(DType.INT8, DType.FLOAT64));
        // exact int64 / uint64: 2^63 - 1 < 2^63, which float64 would round to equal
        NDArray s = NJ.array(new long[]{Long.MAX_VALUE, -1});
        NDArray u = NJ.zeros(DType.UINT64, 2);
        u.setLong(0, Long.MIN_VALUE);  // 2^63
        u.setLong(1, -1L);             // 2^64 - 1
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, true}), NJ.less(s, u)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, true}), NJ.greater(u, s)));
        // out-of-range long scalars compare exactly instead of failing
        NDArray b = NJ.array(DType.UINT8, new double[]{1, 255});
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, true}), NJ.less(b, 300L)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, true}), b.greater(-1L)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, true}), NJ.greater(u, -1L)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, false}), Ufuncs.EQUAL.apply(1000L, b)));
    }

    @Test
    void logicalAndBitwise() {
        NDArray f = NJ.array(new double[]{0, Double.NaN, 2});
        NDArray i = NJ.array(new long[]{1, 1, 0});
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, true, false}), NJ.logicalAnd(f, i)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{true, false, false}), NJ.logicalNot(f)));
        assertEquals(DType.BOOL, Ufuncs.LOGICAL_XOR.resultType(DType.FLOAT32, DType.INT8));
        NDArray c = NJ.array(new double[]{0, 1}).astype(DType.COMPLEX128);
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, true}), NJ.logicalOr(c, c)));
        assertTrue(NJ.arrayEquals(NJ.array(new boolean[]{false, true}), c.astype(DType.BOOL)));
        assertEquals(8, NJ.bitwiseAnd(NJ.array(DType.UINT8, new double[]{12}), 10L).getLong(0));
        assertEquals(DType.INT16, NJ.bitwiseOr(NJ.array(DType.INT8, new double[]{1}),
                NJ.array(DType.UINT8, new double[]{2})).dtype());
        assertEquals(-6, NJ.invert(NJ.array(DType.INT16, new double[]{5})).getLong(0));
        assertTrue(NJ.invert(NJ.array(new boolean[]{false})).getBoolean(0));
        assertThrows(DTypeException.class, () -> NJ.bitwiseXor(NJ.zeros(2), NJ.zeros(2)));
        assertThrows(DTypeException.class, () -> NJ.invert(NJ.zeros(2)));
    }

    @Test
    void unaryArithmetic() {
        NDArray x = NJ.array(new double[]{-0.0, -2.5, Double.NaN, 3});
        assertEquals(0, Double.compare(0.0, NJ.abs(x).getDouble(0)));
        assertEquals(0, Double.compare(0.0, x.sign().getDouble(0)));
        assertEquals(-1.0, x.sign().getDouble(1));
        assertTrue(Double.isNaN(NJ.sign(x).getDouble(2)));
        assertEquals(6.25, x.square().getDouble(1));
        assertEquals(2.5, x.negative().getDouble(1));
        assertEquals(-128, NJ.abs(NJ.array(DType.INT8, new double[]{-128})).getLong(0));
        assertEquals(255, NJ.negative(NJ.array(DType.UINT8, new double[]{1})).getLong(0));
        assertEquals(DType.INT8, Ufuncs.SQUARE.resultType(DType.BOOL));
        assertEquals(DType.BOOL, Ufuncs.ABSOLUTE.resultType(DType.BOOL));
        NDArray b = NJ.array(new boolean[]{true});
        assertThrows(DTypeException.class, () -> NJ.negative(b));
        assertThrows(DTypeException.class, () -> NJ.positive(b));
        assertThrows(DTypeException.class, () -> NJ.sign(b));
        assertEquals("<ufunc 'absolute'>", Ufuncs.ABSOLUTE.toString());
    }

    @Test
    void floatingPointFunctions() {
        // NumPy: np.sqrt of bool/int8/uint8 is float16, of int16 float32, of int32/int64 float64
        assertEquals(DType.FLOAT16, Ufuncs.SQRT.resultType(DType.BOOL));
        assertEquals(DType.FLOAT16, Ufuncs.EXP.resultType(DType.INT8));
        assertEquals(DType.FLOAT16, Ufuncs.LOG.resultType(DType.UINT8));
        assertEquals(DType.FLOAT32, Ufuncs.SIN.resultType(DType.INT16));
        assertEquals(DType.FLOAT64, Ufuncs.COS.resultType(DType.INT32));
        assertEquals(DType.FLOAT64, Ufuncs.TANH.resultType(DType.UINT64));
        assertEquals(DType.BFLOAT16, Ufuncs.SQRT.resultType(DType.BFLOAT16));
        assertEquals(3.0, NJ.sqrt(NJ.array(new double[]{9})).getDouble(0));
        assertEquals(Double.NEGATIVE_INFINITY, NJ.array(new double[]{0}).log().getDouble(0));
        assertTrue(Double.isNaN(NJ.log(NJ.array(new double[]{-1})).getDouble(0)));
        assertEquals(1.0, NJ.exp(NJ.zeros(1)).getDouble(0));
        assertEquals(0.0, NJ.sin(NJ.zeros(1)).getDouble(0));
        assertEquals(1.0, NJ.zeros(1).cos().getDouble(0));
        assertEquals(1.0, NJ.tanh(NJ.array(new double[]{100})).getDouble(0));
        assertThrows(DTypeException.class, () -> NJ.sin(NJ.zeros(DType.COMPLEX64, 1)));
    }
}
