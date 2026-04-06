# Ryzen expression benchmark provenance

The supplied `m3-ryzen-user.json` was reported as measured on a Ryzen 9 5950X
with 128 GB RAM. The file records JMH 1.37 and OpenJDK 25.0.4.1, three forks,
three warmups, five measurements and one thread. Its source revision, operating
system, SMT and governor settings and actual eager provider are unknown. These
results are useful diagnostic evidence, not a completed reference campaign.

`python3 numjvm-bench/run_m3_reference.py --mode scalar --output /tmp/m3-scalar`
runs the original expression benchmark and records a manifest. Repeat with
`--mode simd --output /tmp/m3-simd`; the latter requires the Vector API. The
provider probe is a separate process and must be interpreted along with its
per-operation execution counters. Inspect JMH fork flags and raw results as well.
Use `--manifest-only` to test provenance collection without running Gradle.

The proposed reference machine is the Ryzen host described above. Its complete
configuration must be recorded before treating it as the designated environment.
As proposed before optimization, the objectives for the specified benchmark are
1.5x generated-out speedup for large arrays, a 1.8x allocating regression guard,
and small/medium AUTO budgets in the 0.5.0 plan. These are unverified targets.
