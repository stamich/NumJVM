package smoke;

import java.nio.file.Files;
import java.nio.file.Path;

import org.numjvm.NJ;
import org.numjvm.io.Npy;
import org.numjvm.ndarray.NDArray;

/**
 * Fails with a non-zero exit code if the published artifacts do not work together.
 */
public final class ReleaseCheck {

    private ReleaseCheck() {
    }

    public static void main(String[] args) throws Exception {
        NDArray a = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        NDArray sum = NJ.add(a, NJ.array(new double[]{1, 2, 3}));
        if (NJ.sum(sum).getDouble() != 33.0 || NJ.max(sum, 0).getDouble(2) != 9.0) {
            throw new AssertionError("ufuncs or reductions are wrong: " + sum);
        }
        // The project's own build directory, not the shared temporary directory, which
        // other local users could write to.
        Path file = Files.createDirectories(Path.of("build", "release-check")).resolve("round-trip.npy");
        Npy.write(file, sum);
        NDArray back = Npy.read(file);
        Files.delete(file);
        if (!NJ.arrayEquals(sum, back) || back.getDouble(1, 2) != 9.0) {
            throw new AssertionError("round trip failed: " + back);
        }
        System.out.println("NumJVM " + NJ.version() + " artifacts OK");
        System.out.println(back);
    }
}
