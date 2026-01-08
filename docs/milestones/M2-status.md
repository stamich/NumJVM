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
