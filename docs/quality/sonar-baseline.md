# SonarCloud baseline (0.2.0)

Snapshot of the main branch on 28 November 2025, taken with
[`scripts/sonar_findings.py`](../../scripts/sonar_findings.py) before any 0.2.0 change. The analysis was SonarCloud's
Automatic Analysis (the `SONAR_TOKEN` secret was not set, so the CI analysis had never run), which also scanned
`release-check/`.

| Reliability | Security | Maintainability | Security review | Code smells | Duplication | Lines of code |
|-------------|----------|-----------------|-----------------|------------:|------------:|--------------:|
| C (6 bugs)  | D (1 vulnerability) | A    | A (0 hotspots)  | 0           | 0.0 %       | 3,652         |

Reliability C means the worst open bug has Major severity; Security D means at least one open vulnerability is
Critical.

## Findings and decisions

Every finding is fixed; none is marked a false positive or accepted.

| # | Rule | Location | Finding | Decision | Task |
|--:|------|----------|---------|----------|------|
| 1 | `javabugs:S2259` | `numjvm-core/…/internal/Factories.java:24` | Possible `NullPointerException` | Fix | Q3 |
| 2 | `javabugs:S2259` | `numjvm-core/…/internal/Factories.java:28` | Possible `NullPointerException` | Fix | Q3 |
| 3 | `javabugs:S2259` | `numjvm-core/…/internal/Factories.java:223` | Possible `NullPointerException` | Fix | Q3 |
| 4 | `javabugs:S2259` | `numjvm-core/…/kernels/ValidationKernels.java:65` | Possible `NullPointerException` | Fix | Q3 |
| 5 | `javabugs:S2259` | `numjvm-io/…/io/NpyReader.java:78` | Possible `NullPointerException` | Fix | Q3 |
| 6 | `java:S2111` | `numjvm-core/…/ndarray/ArrayFormatter.java:376` | `new BigDecimal(double)` | Fix | Q3 |
| 7 | `java:S5443` | `release-check/…/smoke/ReleaseCheck.java:22` | Temporary file in a publicly writable directory | Fix | Q4 |

### 1–5: `SharedSecrets` accessors that may return `null`

All five calls dereference `SharedSecrets.arrayAccess()` or `SharedSecrets.scopeAccess()`. Each accessor reads a
`volatile` field that `NDArray` or `NJScope` sets in its static initialiser; if the field is still `null`, the
accessor initialises that class and reads the field again, but returns the second read unchecked. The JVM's class
initialisation rules make a second `null` impossible in practice, but the analyser cannot see that, and nothing in
the code states it. The fix makes the invariant explicit: the accessors throw `IllegalStateException` instead of
returning `null`, so they never return `null` at all and every caller is safe.

### 6: `new BigDecimal(double)` in the float formatter

The constructor is deliberate here: the formatter needs the *exact* decimal value of a float16 or bfloat16 number
before rounding it to the shortest repr, and `BigDecimal.valueOf(double)` would round it to 17 digits first. The fix
keeps the exact value but builds it from the IEEE 754 fields (significand × 2^exponent) in integer arithmetic, which
states the intent and avoids the rule.

### 7: temporary file in `release-check`

`Files.createTempFile` in the shared temporary directory; another local user could pre-create or read the file. The
check now writes into its own `build/` directory. `release-check/` is outside the analysis scope (D1), so this finding
disappears with CI-based analysis anyway; it is fixed because the fix is one line.

### Dependency verification

CI-based analysis also inspects the build scripts and reports dependencies resolved without verification
(`kotlin:S6474`, `text:S8569`). `gradle/verification-metadata.xml` records the SHA-256 checksum of every artifact and
plugin the build resolves, and Gradle refuses an artifact whose checksum differs. After adding or upgrading a
dependency, regenerate it with `./gradlew --write-verification-metadata sha256 build` and review the diff.

## Security review

No security hotspots were open. The rules that raise hotspots in the analysed code (file and network access,
regular expressions, weak randomness, process execution) were checked by hand against `numjvm-io`, which is the
only module that touches files: `.npy` reading enforces `NpyLimits` and parses headers with a restricted literal
parser, and nothing executes code from data.

## Expected result

With all seven findings fixed, the expected ratings are Reliability A, Security A and Maintainability A. They are
confirmed by the first CI-based analysis after `SONAR_TOKEN` is added; from then on the quality gate (Q5) keeps
new code at that level.
