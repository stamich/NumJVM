# 0.3.0-alpha candidate checklist

This checklist prepares a pull request; it does not authorize a tag or publication.

- [x] Candidate version, changelog and roadmap agree on the implemented surface.
- [x] Scalar-only and installed-but-disabled core/oracle tests pass locally.
- [x] Required-SIMD full build passes; direct tests prove accelerated dispatch.
- [x] Java, forced OpenBLAS and automatic linalg contracts pass locally.
- [x] Four library coverage floors pass; native linalg lines remain included.
- [x] Classpath/JPMS published-artifact checks pass with and without optional SIMD.
- [x] Installed-but-unresolved classpath provider falls back automatically.
- [x] M1 fixtures and 24 matrix cases reproduce with pinned NumPy.
- [x] NumPy reads 4,245 NumJVM-produced files with no failures.
- [x] Examples execute in scalar/SIMD and Java/native modes.
- [x] Raw three-fork matrix results, uncertainty and source metadata are retained.
- [x] Commit identities, task boundaries and requested historical dates are checked.
- [ ] Designated-reference performance campaign and small-array budgets approved.
- [ ] Remote build/oracle/backend/Sonar runs inspected on the PR commit.
- [ ] Required branch checks and review confirmed by the maintainer.
- [ ] Tag and package publication after maintainer review (outside this PR task).

See [machine-readable local validation](M2-validation.json), [status](M2-status.md),
[performance evidence](../benchmarks/M2.md), and [retrospective](M2-retrospective.md).
The recorded XML contains 6,400 passing test executions and seven mode-specific
skips across the final required-SIMD build and three linalg modes. All 26 linalg
test cases execute in an applicable mode; auto-only and force-only checks are not
misrepresented as executing in other modes. Property trials run inside their
reported test cases rather than being counted as separate JUnit cases.

Git transport and API access are separate: API CONNECT is policy-blocked. A draft
network addition for api.github.com has been saved, not applied or published.
Until API access is available, PR creation and remote-check inspection are blocked.
