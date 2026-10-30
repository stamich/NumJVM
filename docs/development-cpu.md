# Developing and running the M2 CPU backends

Use JDK 25 and the checked-in Gradle wrapper. No Python or native dependency is
needed for Java development. Each cloud task is already isolated: use the existing
checkout and do not create a worktree unless requested. In the prepared cloud
machine source `/workspace/setup/numjvm-env.sh` before Gradle commands to activate
JDK 25, the writable cache and managed proxy trust. Preserve TLS verification.

```sh
./gradlew build
./gradlew test -PsimdEnabled=true -PsimdMode=required
./gradlew test -PsimdEnabled=true -PsimdMode=off
./gradlew :examples:accelerated-arrays:run
./gradlew :examples:accelerated-arrays:run -PwithoutVector
./gradlew :examples:matmul:run
bash scripts/check_consumers.sh
```

The last command installs only to Maven Local and checks independent classpath and
JPMS consumers, with and without the optional provider. It does not upload packages.
For an application, add `io.github.stamich:numjvm-vector:0.3.0-alpha` at runtime and
launch with `--add-modules=jdk.incubator.vector`. Module-path users may also explicitly
resolve `org.numjvm.vector`. Core-only applications need neither flag nor artifact.
Set `-Dnumjvm.simd=off` to disable selection, or `required` to fail if unavailable.
Unsupported operations retain scalar fallback even in required mode.

## OpenBLAS

The native baseline is Linux x86-64, OpenBLAS LP64 (32-bit dimensions), with CBLAS
and `openblas_get_config`. On Ubuntu/Debian use the signed distribution package
`libopenblas0-pthread`. Other vendors/ILP64 builds are not interchangeable.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./gradlew :numjvm-linalg:backendCheck
OPENBLAS_NUM_THREADS=1 ./gradlew :examples:matmul:run -PlinalgBackend=openblas \
  -PopenblasPath=/absolute/path/to/libopenblas.so.0
```

The cloud validation used OpenBLAS 0.3.29 extracted from Debian's verified package
into `/workspace/toolchains/openblas/`; system installation was not changed.
Pass its full library path using `-PopenblasPath` when reproducing native tests.
Never rename an ILP64 library to imitate an LP64 filename.

For standalone classpath applications use `--enable-native-access=ALL-UNNAMED`;
on the module path use `--enable-native-access=org.numjvm.linalg`.
`-Dnumjvm.linalg=java` forces portability, `auto` allows pre-execution selection,
and `openblas` requires provider availability. Explicit library selection uses
`-Dnumjvm.openblas.path=/absolute/path/to/library`. Backend settings initialize once;
use separate JVMs for mode comparisons. Applications configure BLAS threads before
startup; NumJVM does not silently reset process-wide thread counts.

Linalg supports float32/64 only, uses exact output dtype/shape, preserves scope
ownership and can pack arbitrary input strides. Packing may allocate full native
matrices. Exceptions after execution begins can leave partial user output.
Diagnostics distinguish availability from actual vector/native execution.

## Python oracle and performance

Use Python 3.13 and the hash-pinned requirements in `numjvm-tests-oracle`:

```sh
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python --only-binary :all: --require-hashes -r numjvm-tests-oracle/requirements.txt
.venv/bin/python numjvm-tests-oracle/generate_cases.py --check
.venv/bin/python numjvm-tests-oracle/generate_matmul_cases.py --check
.venv/bin/python numjvm-tests-oracle/verify_results.py numjvm-io/build/oracle-out
```

See [benchmark protocol](benchmarks/M2-protocol.md) for allocating/out separation,
three-fork measurements and uncertainty. Shared cloud measurements do not establish
final performance release gates. See [M2 status](milestones/M2-status.md) before
claiming the release is ready.
