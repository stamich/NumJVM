package org.numjvm.examples;

import java.nio.file.Path;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.io.Npy;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;

/**
 * A tour of NumJVM: create, view, broadcast, compute, reduce, save and load.
 */
public final class BasicArrays {

    private BasicArrays() {
    }

    public static void main(String[] args) {
        // 1. Create an array
        NDArray matrix = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        print("matrix", matrix);
        System.out.println("shape " + matrix.shape() + ", strides " + matrix.strides() + ", dtype " + matrix.dtype().typeName());

        // 2. Zero-copy transposition: only shape and strides change
        NDArray transposed = matrix.transpose();
        print("matrix.transpose()", transposed);
        System.out.println("strides " + transposed.strides() + ", shares memory: " + transposed.sharesMemoryWith(matrix));

        // 3. Reshape (a view here, because the matrix is contiguous)
        NDArray reshaped = matrix.reshape(3, 2);
        print("matrix.reshape(3, 2)", reshaped);

        // 4. Select a sub-array: matrix[:, 0:2] and matrix[::-1]
        print("matrix[:, 0:2]", matrix.slice(Slice.all(), Slice.range(0, 2)));
        print("matrix[::-1]", matrix.slice(Slice.step(-1)));

        // 5. Broadcasting: a (1, 3) row repeated to (2, 3) with stride 0, read-only
        NDArray vector = NJ.array(new double[]{10, 20, 30}, 1, 3);
        NDArray broadcasted = vector.broadcastTo(2, 3);
        print("vector.broadcastTo(2, 3)", broadcasted);
        System.out.println("strides " + broadcasted.strides() + ", writeable: " + broadcasted.isWriteable());
        print("matrix + vector", matrix.add(vector));

        // Element-wise ufuncs and reductions; integers promote as in NumPy
        print("NJ.sqrt(matrix)", NJ.sqrt(matrix));
        print("matrix * 2 (int64 stays int64)", NJ.arange(0, 6, 1).reshape(2, 3).multiply(2L));
        print("matrix > 3", NJ.greater(matrix, 3.0));
        System.out.println("sum(matrix) = " + NJ.sum(matrix).getDouble() + ", mean = " + NJ.mean(matrix).getDouble());
        print("sum along axis 0", NJ.sum(matrix, 0));
        print("max along axis 1, keepdims", NJ.max(matrix, new int[]{1}, true));
        print("int8 + float32 -> " + NJ.resultType(DType.INT8, DType.FLOAT32).typeName(),
                NJ.add(NJ.arange(0, 3, 1).astype(DType.INT8), NJ.array(new float[]{0.5f, 0.25f, 0.125f})));
        print("complex", NJ.exp(NJ.array(new org.numjvm.dtype.Complex[]{org.numjvm.dtype.Complex.of(0, Math.PI)})));

        // 6. Save and 7. load in NumPy's .npy format
        Path file = Path.of("matrix.npy");
        Npy.write(file, matrix);
        NDArray loaded = Npy.read(file);
        System.out.println();
        System.out.println("saved and loaded " + file.toAbsolutePath() + ": equal = " + NJ.arrayEquals(matrix, loaded));

        // Other dtypes and conversions
        print("NJ.arange(0, 12, 1).reshape(3, 4).astype(DType.INT8)", NJ.arange(0, 12, 1).reshape(3, 4).astype(DType.INT8));
        print("NJ.full(DType.FLOAT16, 0.1, 3)", NJ.full(DType.FLOAT16, 0.1, 3));
    }

    private static void print(String label, NDArray a) {
        System.out.println();
        System.out.println(label + " =");
        System.out.println(a);
    }
}
