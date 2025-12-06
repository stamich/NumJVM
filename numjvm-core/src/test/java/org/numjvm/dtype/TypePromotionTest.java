package org.numjvm.dtype;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.numjvm.dtype.DType.BFLOAT16;
import static org.numjvm.dtype.DType.BOOL;
import static org.numjvm.dtype.DType.COMPLEX128;
import static org.numjvm.dtype.DType.COMPLEX64;
import static org.numjvm.dtype.DType.FLOAT16;
import static org.numjvm.dtype.DType.FLOAT32;
import static org.numjvm.dtype.DType.FLOAT64;
import static org.numjvm.dtype.DType.INT16;
import static org.numjvm.dtype.DType.INT32;
import static org.numjvm.dtype.DType.INT64;
import static org.numjvm.dtype.DType.INT8;
import static org.numjvm.dtype.DType.UINT16;
import static org.numjvm.dtype.DType.UINT32;
import static org.numjvm.dtype.DType.UINT64;
import static org.numjvm.dtype.DType.UINT8;

import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.exception.DTypeException;

/**
 * Representative cases; the oracle checks the complete NumPy table.
 */
class TypePromotionTest {

    private static void promotes(DType a, DType b, DType expected) {
        assertEquals(expected, TypePromotion.resultType(a, b), a + " + " + b);
        assertEquals(expected, TypePromotion.resultType(b, a), b + " + " + a);
    }

    @Test
    void withinAKindTheWiderWins() {
        promotes(INT8, INT64, INT64);
        promotes(UINT16, UINT32, UINT32);
        promotes(FLOAT16, FLOAT64, FLOAT64);
        promotes(COMPLEX64, COMPLEX128, COMPLEX128);
    }

    @Test
    void signedWithUnsignedNeedsTheNextSignedWidth() {
        promotes(INT8, UINT8, INT16);
        promotes(INT16, UINT8, INT16);
        promotes(INT8, UINT16, INT32);
        promotes(INT32, UINT32, INT64);
        promotes(INT64, UINT32, INT64);
        promotes(INT64, UINT64, FLOAT64);
        promotes(INT8, UINT64, FLOAT64);
    }

    @Test
    void boolGivesWayToEverything() {
        for (DType t : DType.values()) {
            promotes(BOOL, t, t);
        }
    }

    @Test
    void integerWithFloat() {
        promotes(INT8, FLOAT16, FLOAT16);
        promotes(UINT8, FLOAT16, FLOAT16);
        promotes(INT16, FLOAT16, FLOAT32);
        promotes(UINT16, FLOAT32, FLOAT32);
        promotes(INT32, FLOAT16, FLOAT64);
        promotes(INT32, FLOAT32, FLOAT64);
        promotes(UINT64, FLOAT32, FLOAT64);
        promotes(INT64, FLOAT64, FLOAT64);
    }

    @Test
    void realWithComplex() {
        promotes(FLOAT64, COMPLEX64, COMPLEX128);
        promotes(FLOAT32, COMPLEX64, COMPLEX64);
        promotes(FLOAT16, COMPLEX64, COMPLEX64);
        promotes(INT16, COMPLEX64, COMPLEX64);
        promotes(INT32, COMPLEX64, COMPLEX128);
        promotes(UINT8, COMPLEX128, COMPLEX128);
    }

    @Test
    void bfloat16FollowsMlDtypes() {
        promotes(BFLOAT16, FLOAT16, FLOAT32);
        promotes(BFLOAT16, INT8, BFLOAT16);
        promotes(BFLOAT16, UINT8, BFLOAT16);
        promotes(BFLOAT16, INT16, FLOAT32);
        promotes(BFLOAT16, FLOAT32, FLOAT32);
        promotes(BFLOAT16, FLOAT64, FLOAT64);
        promotes(BFLOAT16, COMPLEX64, COMPLEX64);
        promotes(BFLOAT16, BOOL, BFLOAT16);
    }

    @Test
    void manyDtypesPromoteIntoTheHighestTypeNumber() {
        assertEquals(INT16, TypePromotion.resultType(INT8, UINT8));
        assertEquals(FLOAT32, TypePromotion.resultType(INT8, UINT8, FLOAT32, BOOL));
        // NumPy: np.result_type('i2', 'u2', 'f4') == float32; left to right would give float64
        assertEquals(FLOAT32, TypePromotion.resultType(INT16, UINT16, FLOAT32));
        assertEquals(FLOAT64, TypePromotion.resultType(TypePromotion.resultType(INT16, UINT16), FLOAT32));
        // np.result_type('i1', 'u2', 'f2') == float32: float16 (type number 23) is the start
        assertEquals(FLOAT32, TypePromotion.resultType(INT8, UINT16, FLOAT16));
        assertEquals(FLOAT64, TypePromotion.resultType(INT64, UINT64, FLOAT16));
        assertEquals(INT8, TypePromotion.resultType(INT8));
        assertThrows(DTypeException.class, TypePromotion::resultType);
        assertEquals(INT16, NJ.resultType(NJ.zeros(INT8, 2), NJ.zeros(UINT8)));
        assertEquals(COMPLEX128, NJ.resultType(INT32, COMPLEX64));
    }

    @Test
    void weakScalars() {
        assertEquals(UINT8, TypePromotion.resultTypeWithLong(UINT8));
        assertEquals(INT64, TypePromotion.resultTypeWithLong(BOOL));
        assertEquals(FLOAT16, TypePromotion.resultTypeWithLong(FLOAT16));
        assertEquals(FLOAT64, TypePromotion.resultTypeWithDouble(INT8));
        assertEquals(FLOAT64, TypePromotion.resultTypeWithDouble(BOOL));
        assertEquals(FLOAT32, TypePromotion.resultTypeWithDouble(FLOAT32));
        assertEquals(COMPLEX64, TypePromotion.resultTypeWithDouble(COMPLEX64));
    }

    @Test
    void castingModes() {
        assertTrue(TypePromotion.canCast(INT8, FLOAT16));
        assertFalse(TypePromotion.canCast(INT16, FLOAT16));
        assertTrue(TypePromotion.canCast(INT64, FLOAT64));
        assertFalse(TypePromotion.canCast(INT64, FLOAT32));
        assertTrue(TypePromotion.canCast(UINT64, FLOAT64));
        assertFalse(TypePromotion.canCast(FLOAT64, COMPLEX64));
        assertTrue(TypePromotion.canCast(FLOAT64, COMPLEX64, Casting.SAME_KIND));
        assertTrue(TypePromotion.canCast(UINT64, INT8, Casting.SAME_KIND));
        assertFalse(TypePromotion.canCast(INT8, UINT8, Casting.SAME_KIND));
        assertFalse(TypePromotion.canCast(FLOAT64, INT64, Casting.SAME_KIND));
        assertFalse(TypePromotion.canCast(COMPLEX64, FLOAT64, Casting.SAME_KIND));
        assertTrue(TypePromotion.canCast(UINT64, FLOAT16, Casting.SAME_KIND));
        assertTrue(TypePromotion.canCast(COMPLEX128, INT8, Casting.UNSAFE));
        assertFalse(TypePromotion.canCast(INT8, INT16, Casting.NO));
        assertTrue(TypePromotion.canCast(INT8, INT8, Casting.EQUIV));
        assertTrue(NJ.canCast(BOOL, UINT8));
        assertFalse(NJ.canCast(INT8, UINT64, Casting.SAFE));
    }

    @Test
    void componentDtypes() {
        assertEquals(FLOAT32, TypePromotion.component(COMPLEX64));
        assertEquals(FLOAT64, TypePromotion.component(COMPLEX128));
        assertThrows(DTypeException.class, () -> TypePromotion.component(FLOAT64));
    }

    // ------------------------------------------------------------------ algebraic properties

    @Property
    void commutativeAndIdempotent(@ForAll DType a, @ForAll DType b) {
        assertEquals(TypePromotion.resultType(a, b), TypePromotion.resultType(b, a));
        assertEquals(a, TypePromotion.resultType(a, a));
    }

    @Property
    void bothOperandsCastSafelyToTheResult(@ForAll DType a, @ForAll DType b) {
        DType r = TypePromotion.resultType(a, b);
        if (a == UINT64 && b == INT64 || a == INT64 && b == UINT64 || r == FLOAT64 && (a.size() == 8 && b.isInteger()
                || b.size() == 8 && a.isInteger())) {
            return;  // int64/uint64 promote to float64, which NumPy also calls safe
        }
        assertTrue(TypePromotion.canCast(a, r), a + " -> " + r);
        assertTrue(TypePromotion.canCast(b, r), b + " -> " + r);
    }

    @Property
    void castingModesAreNested(@ForAll DType a, @ForAll DType b) {
        boolean no = TypePromotion.canCast(a, b, Casting.NO);
        boolean safe = TypePromotion.canCast(a, b, Casting.SAFE);
        boolean sameKind = TypePromotion.canCast(a, b, Casting.SAME_KIND);
        assertTrue(!no || safe);
        assertTrue(!safe || sameKind);
        assertTrue(TypePromotion.canCast(a, b, Casting.UNSAFE));
        assertTrue(TypePromotion.canCast(a, a, Casting.SAFE));
    }
}
