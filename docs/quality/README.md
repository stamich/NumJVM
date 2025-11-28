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
