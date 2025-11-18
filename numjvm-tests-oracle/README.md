# NumPy oracle

NumPy is NumJVM's reference implementation. This directory holds the generator, the
committed fixtures, and the script that checks NumJVM's output with NumPy.

| File | Purpose |
| --- | --- |
| `generate_cases.py` | Builds input arrays, runs every case through NumPy, writes `fixtures/` |
| `fixtures/manifest.json` | One entry per case: id, compatibility tag, operation, inputs, parameters, expected result |
| `fixtures/in_*.npy`, `fixtures/expected_*.npy` | Inputs and expected array results |
| `verify_results.py` | Loads the arrays NumJVM wrote and compares them with NumPy's results |

Every case carries one tag:

- `ARRAY_API`: behaviour defined by the Python Array API Standard.
- `NUMPY_COMPATIBILITY`: NumPy behaviour beyond the standard that NumJVM follows
  (printing, contiguity flags, unsafe casts).
- `NUMJVM_SPECIFIC`: a deliberate difference, checked against NumJVM's own rule.

## Workflow

`./gradlew test` never needs Python: `OracleTest` in `numjvm-io` replays the committed
manifest with the JDK alone. The `oracle-tests` CI workflow additionally:

1. installs NumPy and runs `generate_cases.py --check`, which regenerates everything
   into a temporary directory and fails if a committed file differs;
2. runs the Java tests, which write NumJVM's array results to `numjvm-io/build/oracle-out`;
3. runs `verify_results.py numjvm-io/build/oracle-out`, so NumPy itself reads every file
   NumJVM wrote.

To add a case, edit `cases()` in `generate_cases.py`, run it, and commit the fixtures. If
the case uses a new operation, add it to `OracleTest.apply`.
