# 0.6.0 reference performance protocol

The reference campaign requires the user's Ryzen 9 5950X and has not been run
in this environment. Do not treat shared-machine smoke checks as scaling or
release evidence. Run scalar and SIMD campaigns separately on an idle machine,
with JDK 25, Python 3 and a clean checkout of the actual candidate revision.
Record the output of `git status --short` and retain the entire result directory.
Each command requires a new empty output directory without spaces.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 numjvm-bench/run_060_reference.py \
  --mode scalar --workers 1,2,4,8,16 --output /tmp/numjvm-060-scalar
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 numjvm-bench/run_060_reference.py \
  --mode simd --workers 1,2,4,8,16 --output /tmp/numjvm-060-simd
```

The runner records revision, checkout status, JVM, CPU, memory, governor, SMT,
thread settings and the exact JMH command. It uses one JMH caller thread,
three forks, three one-second warmups and five one-second measurements. The
benchmark's own worker count is the `workers` parameter. Each mode runs 120
parameter/method combinations and can take an hour or more. Use `--dry-run`
to inspect the command. A separate optional 32-worker campaign requires a
third output directory and `--workers 32`.

Preserve `manifest.json`, `jmh.log` and `results.json`. The log includes
`NUMJVM_PARALLEL_DECISION`, provider and worker count from setup; verify the
intended paths were actually selected. `JAVA_TOOL_OPTIONS` is recorded, so
inspect it for private information before sharing. Native BLAS thread settings
are not changed by NumJVM; report any external overrides. Compare the parallel
and sequential methods only at matching parameters and report error bars,
small-array overhead, crossover and memory-bandwidth saturation. A single
JMH caller is distinct from the requested NumJVM worker count.

The current shared-machine smoke run used one fork, no warmup and 100 ms
measurements solely to check that ADD, CHAIN and SUM benchmarks execute at one
and two workers. Its timings are not suitable for performance conclusions.
The 0.5.0 fusion, allocating and AUTO targets remain open pending the
reference reruns listed in [performance status](M4-performance-status.md).
