# M2 implementation evidence

Baseline: `2aeff90a1d68f405ed6b635c7de929015ba54af9` (0.2.0-alpha), Java 25,
Gradle wrapper 9.2.0. ADR-001 through ADR-005 are occupied.

The implementation follows [the M2 plan](M2-plan.md), with three independently
validated checkpoints: scalar improvements, optional SIMD, and portable matmul
with optional OpenBLAS. LAPACK and worker-pool execution move to follow-up milestones.

Commit author/committer timestamps are maintainer-requested historical timestamps
in Europe/Warsaw (+01:00). They are not measurement dates or actual effort records.
Validation during this task takes place on 30 October 2026.

## External evidence

- GitHub Git read access works. GitHub API access returned `Forbidden`; remote
  check runs, branch protection, Sonar settings and PR creation are not verified.
- Dependency maintenance is separate from M2; pinned dependencies are retained.
- Performance measurements in this cloud environment are exploratory. No dedicated
  reference machine has been designated, so final performance gates remain open.
- A release tag or package publication must follow maintainer review of evidence.
  This implementation task prepares a PR; it does not publish a release.

## Implementation audit

- Optional vector provider and generated arithmetic/comparison/extrema/unary kernels
  execute without exposing incubator types through core. Exact floating sum keeps
  the logical tree; unsupported runs retain scalar behavior.
- Binary mixed conversion scratch is bounded and released; widening reductions and
  integer mean avoid full-input conversions, including virtual cast-layout mean.
  Compatible real reduction outputs write directly after lifecycle validation.
- Java matmul covers the entire advertised real dtype/shape/stride/batch surface.
  OpenBLAS LP64 float32/64 bindings execute with explicit native-access and library
  configuration. Packing, alias copyback and result scopes have independent tests.
- Scalar-only, installed-but-disabled and required-vector modes are locally exercised.
  Java, forced-native and automatic linalg jobs exercise provider selection separately.
  CI definitions are committed; remote workflow execution has not been observed.
- Four library modules enforce 80% line coverage. Linalg measures combined Java/native
  coverage through `backendCheck`; native source is not excluded to pass a Java-only job.
- Independent classpath/JPMS consumer builds pass with and without the provider. A
  classpath consumer also validates fallback with an unresolved incubator module.
- Native allocation probe: 100 mixed output calls, 100 temporary leases, zero retained
  native bytes. Full matrix JMH samples have three forks and retained uncertainty.

## Outstanding release gates

The [performance report](../benchmarks/M2.md) does not complete the reference-machine
campaign. The 1M-element and same-BLAS 1024² targets, small-array budgets and broad
layout regression evidence remain open. No targets are marked achieved or waived.
GitHub API access is denied at CONNECT; creating the PR and reading remote CI/Sonar
checks requires API access. Git transport is verified separately. The branch is a
review candidate, not evidence of a published 0.3.0-alpha release.
