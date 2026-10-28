# NumJVM

[![build](https://github.com/stamich/NumJVM/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/stamich/NumJVM/actions/workflows/build.yml)
[![NumPy oracle](https://github.com/stamich/NumJVM/actions/workflows/oracle-tests.yml/badge.svg?branch=main)](https://github.com/stamich/NumJVM/actions/workflows/oracle-tests.yml)
[![sonar](https://github.com/stamich/NumJVM/actions/workflows/sonar.yml/badge.svg?branch=main)](https://github.com/stamich/NumJVM/actions/workflows/sonar.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=stamich_NumJVM&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=stamich_NumJVM&metric=coverage)](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=stamich_NumJVM&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=stamich_NumJVM&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=stamich_NumJVM&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=stamich_NumJVM&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Numerical computing for the Java Virtual Machine: N-dimensional arrays with NumPy semantics, built on the Java 25
Foreign Function & Memory API. No Python, NumPy or native libraries are needed at run time.

> **Status: 0.2.0-alpha (milestone M1).** On top of M0's array model, memory management,
> views, broadcasting and `.npy` I/O, NumJVM now has an elementwise engine: 41 ufuncs over
> every dtype, NumPy 2 type promotion, axis reductions, `out=` and in-place operations, and
> complex arithmetic. Everything is tested against NumPy. The API may still change (see
> [`ROADMAP.md`](ROADMAP.md)).

## Quick start

```java
import java.nio.file.Path;

import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.io.Npy;
import org.numjvm.memory.NJScope;
import org.numjvm.ndarray.NDArray;
import org.numjvm.ndarray.Slice;

NDArray m = NJ.array(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
System.out.println(m);
// array([[1., 2., 3.],
//        [4., 5., 6.]])

NDArray t = m.transpose();                         // zero-copy: strides [24, 8] -> [8, 24]
NDArray r = m.reshape(3, 2);                       // a view when the layout allows it
NDArray s = m.slice(Slice.all(), Slice.step(-1));  // m[:, ::-1]
double x = m.getDouble(1, 2);                      // 6.0, no allocation

NDArray b = NJ.array(new double[]{10, 20, 30});
NDArray sum = m.add(b);                            // broadcasting: (2, 3) + (3,)
NDArray big = NJ.greater(m, 3.0);                  // bool array
NDArray e = NJ.exp(m.transpose());                 // any layout; the result keeps it
NDArray cols = NJ.sum(m, 0);                       // [5., 7., 9.]
double mean = NJ.mean(m).getDouble();              // 3.5
NDArray i8 = NJ.arange(0, 6, 1).astype(DType.INT8);
DType p = NJ.add(i8, NJ.zeros(DType.FLOAT32, 6)).dtype();   // float32, as NumPy 2
m.multiplyInPlace(2.0);                            // out= m

Npy.write(Path.of("m.npy"), m);                    // NumPy reads it with np.load
NDArray back = Npy.read(Path.of("m.npy"));

try (NJScope scope = NJ.scope()) {                 // deterministic memory release
    NDArray tmp = scope.zeros(DType.FLOAT64, 1000, 1000);
    NDArray tmp2 = NJ.sqrt(tmp);                   // results of scoped operands stay in the scope
    NDArray kept = scope.keep(tmp2.transpose());   // copy that outlives the scope
}
```

Runnable versions: [`examples/basic-arrays`](examples/basic-arrays) and
[`examples/memory-scopes`](examples/memory-scopes):

```bash
./gradlew :examples:basic-arrays:run
./gradlew :examples:memory-scopes:run
```

## What 0.2 provides

| Area           | Contents                                                                                                         |
|----------------|------------------------------------------------------------------------------------------------------------------|
| Dtypes         | bool, int8–64, uint8–64, float16, bfloat16, float32, float64, complex64/128 (`Complex` values)                   |
| Memory         | heap (`long[]`) and 64-byte aligned native buffers, `NJScope` (confined or shared), `keep()`, `NJ.memoryStats()` |
| Arrays         | shape, byte strides (negative and zero allowed), 0-D and empty arrays, invariants checked once per array         |
| Element access | `getDouble/getLong/getBoolean/getComplex` and setters, rank-specialised, allocation-free                         |
| Views          | `transpose`, `permuteDims`, `swapAxes`, `slice` (NumPy basic indexing), `reshape`/`reshapeView`, `broadcastTo`, `real`/`imag` |
| Copies         | `copy(Order)`, `astype(DType)`                                                                                   |
| Ufuncs         | arithmetic, comparisons, logical and bitwise, `abs`, `sign`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, rounding: 41 in `Ufuncs`, with `NJ` and `NDArray` shortcuts |
| Promotion      | NumPy 2 / Array API 2025.12 lattice, `NJ.resultType`, `NJ.canCast`, weak Java `long`/`double`/`Complex` scalars  |
| Reductions     | `sum` (pairwise), `prod`, `max`, `min`, `mean`, `any`, `all` over axis lists with `keepdims`; `ufunc.reduce`      |
| Output         | `out=` arrays with overlap detection, `addInPlace` and friends; results in the operands' memory order and scope  |
| Iteration      | `IterationPlan` (axis reordering and coalescing), `StridedIterator`, `IndexIterator`                              |
| I/O            | `.npy` read, write and memory-map (format 1.0–3.0, both byte orders, C and Fortran order)                        |
| Printing       | `toString()` matches NumPy's `repr`                                                                              |

## Modules

| Module                | Artifact                        | Contents                                                |
|-----------------------|---------------------------------|---------------------------------------------------------|
| `numjvm-core`         | `io.github.stamich:numjvm-core` | dtypes, memory, NDArray, views, ufuncs, reductions      |
| `numjvm-io`           | `io.github.stamich:numjvm-io`   | NumPy `.npy` I/O                                        |
| `numjvm-bench`        | not published                   | JMH benchmarks                                          |
| `numjvm-tests-oracle` | not published                   | NumPy differential-test fixtures                        |

## Building

Requires JDK 25. The Gradle wrapper downloads Gradle, and the build can provision a JDK 25 toolchain if none is
installed.

```bash
./gradlew build                 # compile, unit, property and oracle tests, coverage
./gradlew :numjvm-bench:jmh -PjmhArgs="UfuncBenchmark -p size=65536 -prof gc"
```

Coverage reports (JaCoCo) are written to `numjvm-*/build/reports/jacoco/test/`. Static analysis runs
on [SonarCloud](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM); see
[`CONTRIBUTING.md`](CONTRIBUTING.md#code-quality-sonarcloud).

The NumPy oracle needs Python only to regenerate or re-verify its fixtures; see
[`numjvm-tests-oracle`](numjvm-tests-oracle/README.md).

## Documentation

- [Architecture](ARCHITECTURE.md) and [design documents](docs/README.md)
- [Architecture decision records](docs/adr/README.md)
- Benchmark results: [M0](docs/benchmarks/M0.md), [M1](docs/benchmarks/M1.md)
- Javadoc: `./gradlew javadoc`, output in `numjvm-*/build/docs/javadoc`

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
