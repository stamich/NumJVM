package org.numjvm.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.numjvm.dtype.DType;
import org.numjvm.ndarray.Shape;

class NpyHeaderTest {

    private static final NpyLimits LIMITS = NpyLimits.defaults();

    /**
     * Builds a file prefix around a header dict, as version {@code major}.
     */
    static byte[] file(int major, String dict) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', (byte) major, 0});
        byte[] text = dict.getBytes(major == 3 ? StandardCharsets.UTF_8 : StandardCharsets.ISO_8859_1);
        int n = text.length;
        out.write(n & 0xFF);
        out.write((n >>> 8) & 0xFF);
        if (major > 1) {
            out.write((n >>> 16) & 0xFF);
            out.write((n >>> 24) & 0xFF);
        }
        out.writeBytes(text);
        return out.toByteArray();
    }

    private static NpyHeader read(byte[] bytes) throws IOException {
        return NpyHeader.read(new ByteArrayInputStream(bytes), LIMITS);
    }

    private static NpyHeader parse(String dict) {
        return NpyHeader.parse(dict, 1, 0, 128, LIMITS);
    }

    @Test
    void parsesTheHeaderNumPyWrites() throws IOException {
        NpyHeader h = read(file(1, "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }" + " ".repeat(58) + "\n"));
        assertEquals(1, h.majorVersion());
        assertEquals(DType.FLOAT64, h.dtype());
        assertEquals(ByteOrder.LITTLE_ENDIAN, h.byteOrder());
        assertFalse(h.fortranOrder());
        assertEquals(Shape.of(2, 3), h.shape());
        assertEquals(48, h.dataBytes());
        assertEquals(128, h.dataOffset());
    }

    @Test
    void versions2And3UseFourByteLengths() throws IOException {
        String dict = "{'descr': '>i4', 'fortran_order': True, 'shape': (4,), }\n";
        NpyHeader v2 = read(file(2, dict));
        assertEquals(2, v2.majorVersion());
        assertEquals(DType.INT32, v2.dtype());
        assertEquals(ByteOrder.BIG_ENDIAN, v2.byteOrder());
        assertTrue(v2.fortranOrder());
        assertEquals(12 + dict.length(), v2.dataOffset());
        assertEquals(3, read(file(3, dict)).majorVersion());
    }

    @Test
    void literalVariants() {
        assertEquals(Shape.scalar(), parse("{'descr': '<f8', 'fortran_order': False, 'shape': ()}").shape());
        assertEquals(Shape.of(3), parse("{\"descr\": \"|u1\", \"fortran_order\": False, \"shape\": (3,)}").shape());
        assertEquals(Shape.of(3, 4), parse("{'shape':(3L,4L),'fortran_order':False,'descr':'<c16'}").shape(),
                "Python 2 headers with long suffixes and no spaces");
        assertEquals(Shape.of(0, 5), parse("  { 'descr' : '=f4' , 'fortran_order' : True , 'shape' : ( 0 , 5 , ) }  \n")
                .shape());
        assertEquals(ByteOrder.nativeOrder(), parse("{'descr': '|b1', 'fortran_order': False, 'shape': (1,)}").byteOrder());
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "",
            "{",
            "{'descr': '<f8', 'fortran_order': False}",                                   // missing key
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'extra': 1}",        // extra key
            "{'descr': '<f8', 'descr': '<f8', 'fortran_order': False, 'shape': (1,)}",    // duplicate
            "{'descr': [('a', '<f8')], 'fortran_order': False, 'shape': (1,)}",           // structured
            "{'descr': '|O', 'fortran_order': False, 'shape': (1,)}",                     // objects (pickle)
            "{'descr': '<U5', 'fortran_order': False, 'shape': (1,)}",                    // strings
            "{'descr': '<f8', 'fortran_order': 0, 'shape': (1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1.5,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (3)}",                     // not a tuple
            "{'descr': '<f8', 'fortran_order': False, 'shape': (99999999999999999999,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296)}", // size overflow
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)} trailing",
            "{'descr': '<f\\x38', 'fortran_order': False, 'shape': (1,)}",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), __import__('os')}",
    })
    void malformedHeadersAreRejected(String dict) {
        assertThrows(NpyFormatException.class, () -> parse(dict), dict);
    }

    @Test
    void badPrefixesAreRejected() {
        assertThrows(NpyFormatException.class, () -> read("PK\u0003\u0004 not npy".getBytes(StandardCharsets.ISO_8859_1)));
        byte[] v4 = file(1, "{}");
        v4[6] = 4;
        assertThrows(NpyFormatException.class, () -> read(v4));
        byte[] minor = file(1, "{}");
        minor[7] = 1;
        assertThrows(NpyFormatException.class, () -> read(minor));
        assertThrows(NpyFormatException.class, () -> read(new byte[]{(byte) 0x93, 'N', 'U'}));
    }

    @Test
    void truncatedHeaderIsRejected() {
        byte[] full = file(1, "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }\n");
        byte[] cut = java.util.Arrays.copyOf(full, full.length - 10);
        NpyFormatException e = assertThrows(NpyFormatException.class, () -> read(cut));
        assertTrue(e.getMessage().contains("truncated"), e.getMessage());
    }

    @Test
    void limitsAreEnforcedBeforeReadingFurther() throws IOException {
        byte[] prefix = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0, 0, 0, 0, 0x7F};  // claims a 2 GiB header
        InputStream in = new ByteArrayInputStream(prefix);
        NpyFormatException e = assertThrows(NpyFormatException.class, () -> NpyHeader.read(in, LIMITS));
        assertTrue(e.getMessage().contains("exceeds the limit"), e.getMessage());

        String dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (1000, 1000), }";
        assertThrows(NpyFormatException.class,
                () -> NpyHeader.parse(dict, 1, 0, 128, LIMITS.withMaxDataBytes(7_999_999)));
        assertEquals(8_000_000, NpyHeader.parse(dict, 1, 0, 128, LIMITS.withMaxDataBytes(8_000_000)).dataBytes());
        assertThrows(NpyFormatException.class, () -> NpyHeader.parse(
                "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 1, 1), }", 1, 0, 128, new NpyLimits(1000, 2, 100)));
        assertThrows(NpyFormatException.class, () -> read(file(1, " ".repeat(10_001))));
        assertThrows(IllegalArgumentException.class, () -> new NpyLimits(0, 64, 1));
    }

    /**
     * Byte-for-byte what NumPy 2's np.save writes for these arrays.
     */
    @Test
    void encodesExactlyLikeNumPy() {
        assertEquals("\u0093NUMPY\u0001\u0000v\u0000{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }"
                        + " ".repeat(58) + "\n",
                new String(NpyHeader.encode("<f8", false, Shape.of(2, 3)), StandardCharsets.ISO_8859_1));
        assertEquals("\u0093NUMPY\u0001\u0000v\u0000{'descr': '<f8', 'fortran_order': False, 'shape': (), }"
                        + " ".repeat(62) + "\n",
                new String(NpyHeader.encode("<f8", false, Shape.scalar()), StandardCharsets.ISO_8859_1));
        assertEquals("\u0093NUMPY\u0001\u0000v\u0000{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }"
                        + " ".repeat(59) + "\n",
                new String(NpyHeader.encode("<f8", true, Shape.of(2, 3)), StandardCharsets.ISO_8859_1));
        assertEquals("\u0093NUMPY\u0001\u0000v\u0000{'descr': '|b1', 'fortran_order': False, 'shape': (1,), }"
                        + " ".repeat(60) + "\n",
                new String(NpyHeader.encode("|b1", false, Shape.of(1)), StandardCharsets.ISO_8859_1));
    }

    @Test
    void encodedHeadersAreAlignedAndReadBack() throws IOException {
        for (Shape s : new Shape[]{Shape.scalar(), Shape.of(7), Shape.of(0, 3), Shape.of(1L << 40, 3),
                Shape.of(new long[64])}) {
            byte[] bytes = NpyHeader.encode("<i2", false, s);
            assertEquals(0, bytes.length % 64, s.toString());
            NpyHeader h = read(bytes);
            assertEquals(s, h.shape());
            assertEquals(bytes.length, h.dataOffset());
            assertArrayEquals(new byte[0], new ByteArrayInputStream(bytes, (int) h.dataOffset(), 0).readAllBytes());
        }
    }
}
