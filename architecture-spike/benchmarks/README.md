# Spike benchmarks (S5)

`src/spike/SpikeBenchmarks.java` is a JMH benchmark over the spike classes in
`../src/spike`. It needs `jmh-core` and `jmh-generator-annprocess` 1.37 (plus their dependencies `jopt-simple` 5.0.4 and
`commons-math3` 3.6.1) on the class path. With the jars in `lib/`:

```bash
cd architecture-spike
CP="lib/*"
javac -cp "$CP" -processorpath "$CP" -d out src/spike/*.java benchmarks/src/spike/*.java
java -cp "out:$CP" org.openjdk.jmh.Main SpikeBenchmarks -prof gc
```

The NumPy baseline runs the same operations at the same sizes:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 benchmarks/numpy_baseline.py
```

From M0.1 on, benchmarks live in the `numjvm-bench` module and run through Gradle. Results are in [
`../results.md`](../results.md).
