package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;

import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.exception.ReadOnlyArrayException;
import org.numjvm.exception.ShapeException;
import org.numjvm.internal.SharedSecrets;
import org.numjvm.testing.ArrayArbitraries;

class TransposeTest {

    private static boolean sameBuffer(NDArray a, NDArray b) {
        return SharedSecrets.arrayAccess().buffer(a) == SharedSecrets.arrayAccess().buffer(b);
    }

    @Test
    void transposeOfAMatrixSwapsShapeAndStrides() {
        NDArray a = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray t = a.transpose();
        assertEquals(Shape.of(3, 2), t.shape());
        assertEquals(Strides.of(8, 24), t.strides());
        assertTrue(sameBuffer(a, t), "zero-copy");
        assertFalse(t.isCContiguous());
        assertTrue(t.isFContiguous());
        assertEquals(4.0, t.getDouble(0, 1));
        t.setDouble(2, 1, 60);
        assertEquals(60.0, a.getDouble(1, 2), "writes through the view are visible in the original");
    }

    @Test
    void transposeOfScalarsAndVectorsIsTheSameLayout() {
        NDArray s = NJ.scalar(1.0).transpose();
        assertEquals(Shape.scalar(), s.shape());
        NDArray v = NJ.arange(5).transpose();
        assertEquals(Shape.of(5), v.shape());
        assertEquals(Strides.of(8), v.strides());
    }

    @Test
    void permuteDimsReordersAxes() {
        NDArray a = NJ.zeros(2, 3, 4);
        NDArray p = a.permuteDims(1, 2, 0);
        assertEquals(Shape.of(3, 4, 2), p.shape());
        assertEquals(Strides.of(32, 8, 96), p.strides());
        assertEquals(Shape.of(4, 3, 2), a.permuteDims(-1, -2, -3).shape());
    }

    @Test
    void invalidPermutationsAreRejected() {
        NDArray a = NJ.zeros(2, 3, 4);
        assertThrows(ShapeException.class, () -> a.permuteDims(0, 1));
        assertThrows(ShapeException.class, () -> a.permuteDims(0, 1, 1));
        assertThrows(ShapeException.class, () -> a.permuteDims(0, 1, 3));
    }

    @Test
    void swapAxes() {
        NDArray a = NJ.zeros(2, 3, 4);
        assertEquals(Shape.of(4, 3, 2), a.swapAxes(0, 2).shape());
        assertEquals(Shape.of(2, 4, 3), a.swapAxes(-1, 1).shape());
        assertSame(a.dtype(), a.swapAxes(0, 0).dtype());
        assertThrows(ShapeException.class, () -> a.swapAxes(0, 3));
    }

    @Test
    void viewsKeepTheReadOnlyFlag() {
        NDArray ro = SharedSecrets.arrayAccess().create(SharedSecrets.arrayAccess().buffer(NJ.zeros(3)),
                org.numjvm.dtype.DType.FLOAT64, 0, Shape.of(2, 3), Strides.of(0, 8), false);
        NDArray t = ro.transpose();
        assertFalse(t.isWriteable());
        assertThrows(ReadOnlyArrayException.class, () -> t.setDouble(0, 0, 1));
    }

    @Provide
    Arbitrary<NDArray> arrays() {
        return ArrayArbitraries.arrays();
    }

    @Property
    void transposingTwiceRestoresShapeAndStrides(@ForAll("arrays") NDArray a) {
        NDArray tt = a.transpose().transpose();
        assertEquals(a.shape(), tt.shape());
        assertEquals(a.strides(), tt.strides());
        assertArrayEquals(ArrayArbitraries.values(a), ArrayArbitraries.values(tt));
    }

    @Property
    void transposeMapsIndicesInReverse(@ForAll("arrays") NDArray a) {
        NDArray t = a.transpose();
        long[] dims = a.shape().dimensions();
        long[] idx = new long[dims.length];
        long[] rev = new long[dims.length];
        for (long n = 0; n < a.size(); n++) {
            for (int d = 0; d < dims.length; d++) {
                rev[dims.length - 1 - d] = idx[d];
            }
            assertEquals(a.getDouble(idx), t.getDouble(rev));
            for (int d = dims.length - 1; d >= 0; d--) {
                if (++idx[d] < dims[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    @Property
    void permutationFollowedByItsInverseIsTheIdentity(@ForAll("arrays") NDArray a, @ForAll java.util.Random rnd) {
        int n = a.ndim();
        java.util.List<Integer> axes = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            axes.add(i);
        }
        java.util.Collections.shuffle(axes, rnd);
        int[] perm = axes.stream().mapToInt(Integer::intValue).toArray();
        int[] inverse = new int[n];
        for (int i = 0; i < n; i++) {
            inverse[perm[i]] = i;
        }
        NDArray back = a.permuteDims(perm).permuteDims(inverse);
        assertEquals(a.shape(), back.shape());
        assertEquals(a.strides(), back.strides());
    }
}
