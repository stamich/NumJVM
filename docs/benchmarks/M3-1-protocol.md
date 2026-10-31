# 0.5.0 reference benchmark protocol

The proposed reference machine is the user's Ryzen 9 5950X with 128 GB RAM.
Record the actual OS, JDK, CPU settings, JVM flags, selected providers, thread
settings, source revision and dirty state before interpreting results. The
previous user-supplied JSON has an unknown revision and provider; preserve it as
historical evidence rather than relabeling it as a completed baseline.

Use JDK 25 and Python 3 on a clean checkout of the candidate revision. Run scalar
and required-SIMD configurations separately. `--suite all` includes the original
float64 baseline, homogeneous/mixed dtypes, allocating/out, C/F, strided and
broadcast layouts, in-place output, lifecycle costs and scratch-byte comparisons.
It performs many three-fork JMH cases and can take several hours per mode.
Each invocation requires a new, empty output directory without whitespace.

```sh
git status --short
git rev-parse HEAD
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 numjvm-bench/run_050_reference.py \
  --mode scalar --suite all --output /tmp/numjvm-050-scalar
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 numjvm-bench/run_050_reference.py \
  --mode simd --suite all --output /tmp/numjvm-050-simd
```

Use `--dry-run` to inspect commands without executing them, or choose one suite
with `--suite baseline|contiguous|layouts|inplace|inplace-generated|lifecycle|scratch`.
The baseline runs first for every selected suite to capture provider provenance.
Actual vector execution is checked by counters; a provider being installed alone
does not establish that the benchmark used it. Inspect the per-suite logs and
JMH JSON, including `NUMJVM_AUTO_DECISION` and `NUMJVM_AUTO_VECTOR_RUNS`.

Keep the resulting directories unchanged. Their manifests include source and
machine details plus `JAVA_TOOL_OPTIONS`; inspect those options before sharing
if they contain private values. The comparison tool rejects missing or
incompatible provenance rather than presenting a misleading ratio. Heap GC
allocation does not include all native storage; use the separate expression
memory probe and preserve its output alongside the JMH results.

Compare `autoOut` against `eagerOut` when AUTO reports `fused-vector`; the
`generatedOut` benchmark uses REQUIRED's scalar-generated contract. Compare
allocating methods only against other allocating methods.

The proposed 1.5× large fused-output target, 1.8× allocating regression guard,
and small/medium AUTO budgets remain open pending valid measurements. If a gate
fails, report it and retain the failure; do not silently lower the target.
