package org.numjvm.internal.loops;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;
import org.numjvm.ndarray.NDArray;

class KernelRegistryTest {

    @Test
    void looksUpLoopsByOperationAndInputDtypes() {
        BinaryLoop add = KernelRegistry.binary("add", DType.FLOAT64, DType.FLOAT64);
        NDArray out = NJ.zeros(3);
        LoopExecutor.binary(add, NJ.array(new double[]{1, 2, 3}), NJ.scalar(0.5), out);
        assertTrue(NJ.arrayEquals(NJ.array(new double[]{1.5, 2.5, 3.5}), out));
        assertNotNull(KernelRegistry.unary("copy", DType.FLOAT64));
    }

    @Test
    void signaturesDescribeTheLoops() {
        LoopSignature sig = KernelRegistry.find("add", DType.FLOAT64, DType.FLOAT64);
        assertNotNull(sig);
        assertEquals("add(float64, float64) -> float64", sig.toString());
        assertEquals(2, sig.arity());
        assertEquals(List.of(DType.FLOAT64, DType.FLOAT64), sig.inputs());
        assertTrue(KernelRegistry.signatures("add").contains(sig));
        assertTrue(KernelRegistry.operations().contains("add"));
        assertNull(KernelRegistry.find("add", DType.COMPLEX64, DType.FLOAT64));
        assertNull(KernelRegistry.find("no-such-op", DType.FLOAT64));
    }

    @Test
    void missingLoopsAreDTypeExceptions() {
        DTypeException e = assertThrows(DTypeException.class,
                () -> KernelRegistry.binary("add", DType.COMPLEX64, DType.FLOAT64));
        assertTrue(e.getMessage().contains("(complex64, float64)"), e.getMessage());
        assertTrue(e.getMessage().contains("available"), e.getMessage());
        assertThrows(DTypeException.class, () -> KernelRegistry.unary("add", DType.FLOAT64));
        assertThrows(DTypeException.class, () -> KernelRegistry.signatures("no-such-op"));
    }

    @Test
    void registrarRejectsDuplicatesAndMixedArity() {
        KernelRegistry.Registrar r = new KernelRegistry.Registrar();
        r.binary("x", DType.INT8, DType.INT8, DType.INT8, (a, pa, sa, b, pb, sb, o, po, so, n) -> { });
        assertThrows(IllegalStateException.class,
                () -> r.binary("x", DType.INT8, DType.INT8, DType.INT8, (a, pa, sa, b, pb, sb, o, po, so, n) -> { }));
        assertThrows(IllegalStateException.class,
                () -> r.unary("x", DType.INT8, DType.INT8, (in, pi, si, o, po, so, n) -> { }));
    }
}
