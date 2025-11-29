# Code quality

NumJVM's library code is analysed on [SonarCloud](https://sonarcloud.io/summary/new_code?id=stamich_NumJVM). The
target for every release from 0.2.0 on is an A rating for Reliability, Security and Maintainability, with every
security hotspot reviewed.

## D1 — Analysis scope

| Analysed as | Paths |
|-------------|-------|
| Main code   | `numjvm-core/src/main/java`, `numjvm-io/src/main/java` |
| Test code   | `numjvm-core/src/test/java`, `numjvm-io/src/test/java` |

Not analysed:

- `numjvm-bench`, `examples/` and `release-check/`: not shipped; they only consume the public API.
- `architecture-spike/`: throwaway code kept as the record of the M-1 measurements.
- `numjvm-tests-oracle/`: Python fixture generators and generated `.npy` files.
- Build scripts and workflows (`sonar.gradle.scanAll=false`).

The scope is set twice, once for each analysis mode:

| Mode | Where the scope is set | Coverage |
|------|------------------------|----------|
| CI-based analysis (`.github/workflows/sonar.yml`) | root `build.gradle.kts`: only `numjvm-core` and `numjvm-io` are analysed | JaCoCo, from every test run |
| Automatic Analysis (SonarCloud's default for a new project) | `.sonarcloud.properties` | none |

CI-based analysis is the intended mode: it analyses compiled classes, so rules that need type information are
precise, and it imports coverage. Automatic Analysis must be switched off before it can run, because SonarCloud
rejects a CI analysis while Automatic Analysis is on.

## Switching on CI-based analysis

1. In SonarCloud, project `stamich_NumJVM`, *Administration → Analysis Method*: switch off Automatic Analysis.
2. Create a token in SonarCloud (*My Account → Security*) and add it to the GitHub repository as the Actions secret
   `SONAR_TOKEN`.
3. Re-run the `sonar` workflow on `main`. The first run analyses the whole code base and becomes the baseline for
   "new code".

Until the secret exists, the `sonar` workflow builds and tests but skips the upload, and the ratings shown on SonarCloud
come from Automatic Analysis.

If the `sonar` workflow fails in its *Analyse* step, the step repeats the scanner's reason as an error annotation on the
run. The usual one is "You are running CI analysis while Automatic Analysis is enabled": the secret is in place but
step 1 above has not been done.

## D2 — Quality gate

NumJVM uses SonarCloud's built-in gate, **Sonar way**, on new code:

| Condition on new code | Threshold |
|-----------------------|-----------|
| Reliability, Security and Maintainability ratings | A |
| Security hotspots reviewed | 100 % |
| Coverage | ≥ 80 % |
| Duplicated lines | ≤ 3 % |

It is enforced in three places:

1. **The `sonar` workflow fails when the gate fails.** The analysis sets `sonar.qualitygate.wait=true`, so
   `./gradlew sonar` waits for SonarCloud's verdict (up to 5 minutes) and exits with an error on a failed gate.
2. **`./gradlew check` fails below 80 % line coverage** of `numjvm-core` or `numjvm-io`
   (`jacocoTestCoverageVerification`), so coverage is also checked locally and in the `build` workflow, which
   reports both modules' coverage on every run.
3. **Branch protection on `main`** makes the checks required. In the repository settings, *Branches → Branch
   protection rules → main*, enable *Require status checks to pass before merging* and select:
   - `Build and test (JDK 25)` and `NumPy oracle`;
   - `SonarCloud analysis` (this workflow's job);
   - `SonarCloud Code Analysis` (the check SonarCloud's GitHub app posts on pull requests).

   This is a repository setting, not a file, so the repository owner sets it once.

## CI hardening

- Every third-party action is pinned to a full commit SHA, with the release in a comment
  (`uses: actions/checkout@<sha>  # v7.0.1`). Updating an action means updating SHA and comment together; there is
  no automated update bot for now.
- Python dependencies of the oracle are pinned with hashes and installed with
  `pip install --only-binary :all: --require-hashes`: no other version, and no source distribution with a setup script.
- All actions run on Node 24; jobs run on a fixed image, `ubuntu-24.04`, so a change of `ubuntu-latest` cannot
  change a build unannounced.
- Least privilege: workflows default to `contents: read`; only the build job may write check runs (test results)
  and only the sonar job may read pull requests.
- `actions/checkout` does not persist the job's token in the working copy (`persist-credentials: false`); no job
  pushes.
- `gradle/actions/setup-gradle` validates the Gradle wrapper checksum before running it.
