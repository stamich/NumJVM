# Contributing to NumJVM

## Build and test

```bash
./gradlew build          # compile, unit and property tests, all modules
./gradlew :numjvm-core:test --tests 'org.numjvm.ndarray.*'
```

JDK 25 is required. `numjvm-core` and `numjvm-io` have no runtime dependencies; keep it that way. Test-only dependencies
are JUnit 5 and jqwik.

## Code rules

- Library code compiles with `-Xlint:all -Werror`. Fix warnings, do not suppress them without a comment explaining why.
- Public types and methods carry Javadoc. Internal packages (`org.numjvm.internal.*`)
  are not exported and may change at any time.
- Every public behaviour that mirrors NumPy states which compatibility profile it follows: `ARRAY_API` (Python Array API
  Standard), `NUMPY_COMPATIBILITY` (NumPy behaviour beyond the standard), or `NUMJVM_SPECIFIC` (deliberate difference).
- No per-element allocation in element access or kernels. Benchmarks with `-prof gc`
  are the check.

## Code quality (SonarCloud)

Every pull request and every push to `main` is analysed on
[SonarCloud](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM) by
[`.github/workflows/sonar.yml`](.github/workflows/sonar.yml). Coverage comes from JaCoCo, which runs after every `test`
task.

Only the main code of `numjvm-core` and `numjvm-io` is analysed, with their tests as test code; see
[`docs/quality`](docs/quality/README.md) for the scope, the one-time setup and the quality gate. New code must pass
the gate (A ratings, every hotspot reviewed, at least 80 % coverage); `./gradlew check` already fails when a library
module drops below 80 % line coverage.

Running an analysis locally:

```bash
SONAR_TOKEN=... ./gradlew build sonar
```

For a self-hosted SonarQube server, set `SONAR_HOST_URL` and pass your keys, e.g.
`./gradlew build sonar -Dsonar.projectKey=numjvm -Dsonar.organization=`.

## Generated inner loops

The scalar inner loops in `org.numjvm.internal.loops` (`Int8Loops`, `Float64Loops`, ...) are generated, not written
by hand (decision D3). To change them, edit the templates in `numjvm-core/src/codegen/templates` or the operation table
in `numjvm-core/src/codegen/java/org/numjvm/codegen/Ops.java`; the `generateKernels` task regenerates them into
`numjvm-core/build/generated/sources/kernels` before every compilation. Generated sources are never committed.

## Decisions

Significant design decisions are recorded as ADRs in [`docs/adr`](docs/adr). Open a new ADR (or supersede an existing
one) in the same pull request that changes the design.

## Commits and pull requests

One logical change per commit, with a subject line in the imperative mood. Pull requests must be green on all CI
workflows (build, NumPy oracle and SonarCloud) and pass the SonarCloud quality gate.
