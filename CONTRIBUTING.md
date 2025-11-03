# Contributing to NumJVM

## Build and test

```bash
./gradlew build          # compile, unit and property tests, all modules
./gradlew :numjvm-core:test --tests 'org.numjvm.ndarray.*'
```

JDK 25 is required. `numjvm-core` and `numjvm-io` have no runtime dependencies; keep
it that way. Test-only dependencies are JUnit 5 and jqwik.

## Code rules

- Library code compiles with `-Xlint:all -Werror`. Fix warnings, do not suppress them
  without a comment explaining why.
- Public types and methods carry Javadoc. Internal packages (`org.numjvm.internal.*`)
  are not exported and may change at any time.
- Every public behaviour that mirrors NumPy states which compatibility profile it
  follows: `ARRAY_API` (Python Array API Standard), `NUMPY_COMPATIBILITY` (NumPy
  behaviour beyond the standard), or `NUMJVM_SPECIFIC` (deliberate difference).
- No per-element allocation in element access or kernels. Benchmarks with `-prof gc`
  are the check.

## Decisions

Significant design decisions are recorded as ADRs in [`docs/adr`](docs/adr). Open a new
ADR (or supersede an existing one) in the same pull request that changes the design.

## Commits and pull requests

One logical change per commit, with a subject line in the imperative mood. Pull
requests must be green on both CI workflows (build and NumPy oracle).
