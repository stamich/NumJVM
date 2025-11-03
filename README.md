# NumJVM

Numerical computing for the Java Virtual Machine: multidimensional arrays with NumPy
semantics, built on the Java 25 Foreign Function & Memory API.

> **Status:** pre-alpha. Milestone M0 (core array and memory foundation) is in
> progress. Design decisions are recorded in [`docs/adr`](docs/adr).

## Modules

| Module | Contents |
| --- | --- |
| `numjvm-core` | dtypes, memory management, `NDArray`, views, broadcasting |
| `numjvm-io` | NumPy `.npy` reading and writing |
| `numjvm-bench` | JMH benchmarks (not published) |

## Building

Requires JDK 25. The Gradle wrapper downloads Gradle; the build can provision a JDK 25
toolchain automatically if none is installed.

```bash
./gradlew build
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
