package org.numjvm.dtype;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteOrder;
import java.util.EnumSet;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.exception.DTypeException;

class DTypeTest {

    @Test
    void allFifteenDTypesExist() {
        assertEquals(15, DType.values().length);
    }

    @ParameterizedTest
    @CsvSource({
            "BOOL, 1, 1, BOOL",
            "INT8, 1, 1, SIGNED_INTEGER",
            "INT16, 2, 2, SIGNED_INTEGER",
            "INT32, 4, 4, SIGNED_INTEGER",
            "INT64, 8, 8, SIGNED_INTEGER",
            "UINT8, 1, 1, UNSIGNED_INTEGER",
            "UINT16, 2, 2, UNSIGNED_INTEGER",
            "UINT32, 4, 4, UNSIGNED_INTEGER",
            "UINT64, 8, 8, UNSIGNED_INTEGER",
            "FLOAT16, 2, 2, FLOATING",
            "BFLOAT16, 2, 2, FLOATING",
            "FLOAT32, 4, 4, FLOATING",
            "FLOAT64, 8, 8, FLOATING",
            "COMPLEX64, 8, 4, COMPLEX",
            "COMPLEX128, 16, 8, COMPLEX",
    })
    void sizeAlignmentAndKind(DType dtype, int size, int alignment, DTypeKind kind) {
        assertEquals(size, dtype.size());
        assertEquals(alignment, dtype.alignment());
        assertEquals(kind, dtype.kind());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void layoutSizeMatchesElementSize(DType dtype) {
        assertEquals(dtype.size(), dtype.layout().byteSize());
        assertEquals(1, dtype.layout().byteAlignment(), "layouts are unaligned by design (ADR-002)");
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, mode = EnumSource.Mode.EXCLUDE, names = "BFLOAT16")
    void numpyDescriptorRoundTrips(DType dtype) {
        String descr = dtype.numpyDescriptor();
        assertEquals(dtype, DType.fromNumpyDescriptor(descr));
        char order = descr.charAt(0);
        if (dtype.size() == 1) {
            assertEquals('|', order);
        } else {
            assertEquals(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? '<' : '>', order);
        }
    }

    @Test
    void bfloat16HasNoNumpyDescriptor() {
        assertFalse(DType.BFLOAT16.hasNumpyDescriptor());
        assertThrows(DTypeException.class, DType.BFLOAT16::numpyDescriptor);
    }

    @ParameterizedTest
    @CsvSource({
            "<f8, FLOAT64", ">f8, FLOAT64", "=f8, FLOAT64", "f8, FLOAT64",
            "|b1, BOOL", "<i8, INT64", ">u2, UINT16", "<c16, COMPLEX128", "<c8, COMPLEX64",
            "<f2, FLOAT16", "|u1, UINT8", "|i1, INT8",
    })
    void parsesNumpyDescriptors(String descriptor, DType expected) {
        assertEquals(expected, DType.fromNumpyDescriptor(descriptor));
    }

    @ParameterizedTest
    @ValueSource(strings = {"", "<", "<f3", "<U5", "|O", "|V8", "<M8[ns]", "<m8", "x", "|f8", "<f16", "bool"})
    void rejectsUnsupportedDescriptors(String descriptor) {
        assertThrows(DTypeException.class, () -> DType.fromNumpyDescriptor(descriptor));
    }

    @Test
    void rejectsNullDescriptor() {
        assertThrows(DTypeException.class, () -> DType.fromNumpyDescriptor(null));
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void namesRoundTrip(DType dtype) {
        assertEquals(dtype, DType.fromName(dtype.typeName()));
        assertEquals(dtype, DType.fromName(dtype.typeName().toUpperCase()));
    }

    @Test
    void unknownNameIsRejected() {
        assertThrows(DTypeException.class, () -> DType.fromName("float128"));
    }

    @Test
    void categoryPredicatesPartitionTheDTypes() {
        for (DType d : DType.values()) {
            int categories = (d.isBool() ? 1 : 0) + (d.isInteger() ? 1 : 0)
                    + (d.isFloatingPoint() ? 1 : 0) + (d.isComplex() ? 1 : 0);
            assertEquals(1, categories, d.toString());
        }
        assertEquals(EnumSet.of(DType.UINT8, DType.UINT16, DType.UINT32, DType.UINT64),
                EnumSet.copyOf(java.util.Arrays.stream(DType.values()).filter(DType::isUnsigned).toList()));
        assertTrue(DType.FLOAT64.isFloatingPoint());
        assertEquals('c', DType.COMPLEX64.kind().numpyKind());
    }
}
