# M2 measurement protocol

Run from the checkout with JDK 25 activated:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/m2_benchmark.py --output build/m2-baseline
```

Use `--smoke` only to validate the runner. It still runs three forks, but its
100 ms intervals cannot establish performance or regression gates. The runner
checks the number of completed sizes/forks and preserves Gradle's failure status.
Store raw JSON, logs and environment metadata together. Never compare smoke
results against full benchmark runs.

Use the same physical machine and JVM settings for the M1 baseline and candidate.
Measure 1,024 / 65,536 / 1,048,576 elements, allocating and preallocated operations,
int32 sum/mean, mixed conversion, C/F/strided/broadcast layouts, and rectangular,
batched and square matrices. Report native retained/allocated bytes separately
from `-prof gc`. Keep BLAS library, version and thread count identical to NumPy.
Measure packing-inclusive and already-packed paths separately.

Final targets from M2-plan section 8.3 remain provisional until a stable reference
machine is designated. Record per-case small-array regression budgets and confidence
intervals before tuning. Shared CI runs correctness and smoke measurements only.
