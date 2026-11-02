# 0.5.0 reference investigation

The user ran these benchmarks on a Ryzen 9 5950X with 128 GB RAM, Linux and
JDK 25.0.4.1. The initial full scalar and SIMD campaigns recorded source
revision `4aff71731d21f8523fd733e853c2961ab41ef235` but a dirty checkout;
they are diagnostic evidence, not proof of the exact source used. The follow-up
C-order and threshold runs started from that revision with an empty recorded
`git status --short`. The protocol uses one JMH worker, three forks, three
one-second warmups and five one-second measurements. The SIMD provider was
`jdk-vector`; the baseline probe verified vector execution. JMH setup also
reports the AUTO decision. The user supplied the full campaign archives, the
follow-up JSON and logs, and four JFR recordings; those raw artifacts are not
included in this repository.

## Representative C-order results

Times are microseconds per operation, lower is better. The follow-up compares
preallocated outputs, float32 inputs and the SIMD provider. These are selected
cases, not a claim about every shape or operation.

| Operation | Elements | eagerOut | autoOut | blockOut | AUTO decision in setup |
|---|---:|---:|---:|---:|---|
| ADD | 256 | 1.258 | 2.248 | 2.277 | eager |
| CHAIN | 256 | 1.431 | 4.464 | 2.498 | eager |
| ADD | 65,536 | 230.579 | 268.140 | 234.377 | fused-vector |
| CHAIN | 65,536 | 113.712 | 250.730 | 125.133 | fused-vector |
| ADD | 1,048,576 | 103.965 | 2,048.067 | 141.559 | fused-vector |
| CHAIN | 1,048,576 | 395.746 | 3,683.991 | 2,052.027 | fused-vector |

For the large ADD and CHAIN cases, JMH reported respectively 16,785,621 and
33,568,252 bytes of heap allocation per AUTO operation, versus approximately
1,089 and 3,267 bytes for eager. The threshold sweep found AUTO slower than
eager for both float32 ADD and CHAIN at every tested power-of-two size from
128 through 8,192. Its setup reported eager through 2,048 and fused-vector at
4,096 and 8,192. Setup decisions are not a trace of every timed invocation.

## Fortran-order correction

The original benchmark allocates C-order eager intermediates even when the
inputs and output are F-order. The follow-up modified only the benchmark to
copy both intermediates to F order, preserving the patch with the results. In
that run the recorded status also listed two untracked JSON files. For 1,048,576
elements, F-order CHAIN eager versus AUTO was 391.303 versus 3,664.833 µs for
float32, and 3,002.260 versus 7,548.619 µs for float64. The earlier apparent
F-order AUTO advantage therefore does not hold in these corrected cases. The
benchmark patch has not yet been applied to this repository.

## JFR and compilation

Four five-second JFR recordings cover C-order float32 ADD and CHAIN, eager and
AUTO, with 1,048,576 elements. AUTO allocation samples predominantly point to
vector loads in `FloatFusedKernel.run`; sampled types include `float[]` and
`Float256Vector`. AUTO recordings contain 27 and 24 garbage collections for
ADD and CHAIN respectively, while the eager recordings contain none. Samples
are not allocation counts. The vector array and per-chunk call to
`expression.evaluate` are a plausible barrier to allocation elimination, but
JFR alone does not prove the underlying compiler cause. Profiled timings and
allocation samples are diagnostic; the three-fork JMH measurements above are
the comparison evidence.

One separate compilation probe reported tracing 2.37 ms, first execution
120.62 ms, instrumented compilation 3.86 ms and a subsequent cache-hit execution
3.26 ms. It is one diagnostic invocation; first execution includes work beyond
compilation. The separate lifecycle JMH suite also ran.

## Gate status and limitations

* The representative large C-order float32 fused-output cases fail to show the
  proposed 1.5× advantage: AUTO is substantially slower than eager. This does
  not determine every case in the full gate matrix.
* In the separate one-million-element float64 baseline, scalar-generated
  allocating execution was 4,115.495 µs versus eager allocating 9,134.181 µs
  in the later lifecycle campaign. This observed ratio exceeds 1.8×, but it
  is one case, from a run whose manifest was dirty, and does not establish the
  complete allocating regression gate or AUTO fused-vector performance.
* The small-input measurements expose AUTO overhead; assess the complete
  budget using its specified cases and absolute limits.
* Dedicated dot/vecdot performance and isolated multi-run compilation evidence
  are not established by these archives. Reference calibration and remaining
  M2/M3/M3.1 performance gates remain open. No release is implied.

See the [protocol](M3-1-protocol.md) and the proposed
[0.6.0 plan](../milestones/M4-0.6-plan.md).
