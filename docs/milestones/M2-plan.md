# NumJVM 0.3.0-alpha — M2 implementation proposal

**Accelerated CPU execution and portable matrix multiplication**

- Proposal date: 29 October 2026
- Status: proposed implementation plan; estimates and API names are provisional
- Repository: [stamich/NumJVM](https://github.com/stamich/NumJVM)
- Analysis baseline: [`2aeff90a1d68f405ed6b635c7de929015ba54af9`](https://github.com/stamich/NumJVM/commit/2aeff90a1d68f405ed6b635c7de929015ba54af9)

## 1. Release objective

M2 should make NumJVM's existing numerical operations materially faster and add useful matrix multiplication while preserving the array, dtype, memory ownership, and fallback behavior established in M0 and M1.

The proposed release contains **48 implementation and delivery tasks**, organized into six workstreams. Each task has a concrete deliverable, acceptance criteria, dependencies, and an effort estimate. Feature tests belong to their implementation tasks; the verification workstream covers interactions between features and external consumers.

The task count follows the deliverables. It is neither a requirement to exceed forty nor a commitment to exactly forty-eight commits. Prefer one reviewable change per task, with additional commits where needed. Squash or split commits according to review needs; record actual commit dates.

The release is **0.3.0-alpha**, because backend configuration and the new linear algebra API remain experimental. This document is a new proposal based on the repository review and the supplied milestone document, rather than an assertion that its design decisions have already been accepted or implemented.

## 2. Evidence and constraints

The reviewed baseline already contains generated scalar ufunc loops, iteration plans, pairwise floating reductions, dtype promotion, scoped memory, NumPy differential tests, and JMH benchmarks. The build, oracle, and Sonar workflow runs inspected for that baseline succeeded. Tests and performance measurements were not independently rerun for this proposal.

| Observation from the baseline | Planning consequence |
| --- | --- |
| The M1 report gives approximately 21.6× NumPy time for int32 sum and 16.9× for int32 mean. Operand conversion materializes a new array. | Implement mixed-input/accumulator reductions before relying on SIMD to fix the cost. |
| Reduction `out=` computes a result and then stores it into the supplied output. | Add direct output execution, with explicit overlap and casting rules. |
| Allocating float64 add at 65,536 elements is about 6.3× NumPy in the report, while preallocated add is about 0.97×. | Measure allocation separately from arithmetic. The 512 KiB result is below the documented 1 MiB heap threshold; native-only changes cannot explain or fix this example by themselves. |
| The reviewed native allocator calls a standard Java arena allocator without an extra explicit fill. | Do not assume a removable second zero-fill exists. Standard arenas return zero-initialized storage. |
| Pairwise sums use specific accumulator and block-combination orders. | Faster reductions must preserve the applicable arithmetic contract; arbitrary horizontal sums are insufficient. |
| Default scopes are confined to the creating thread. | General worker-pool execution needs a separate ownership design and is deferred. |
| Existing performance data comes from shared hardware and includes noisy native-allocation measurements. | Establish a reproducible baseline before setting final performance thresholds. |

Sources are listed in Section 13. Historical benchmark ratios are motivation, not forecasts for different hardware or claims that every timing difference has one proven cause.

## 3. Committed scope and exclusions

### 3.1 Included in M2

- Bounded scratch storage for selected mixed-dtype elementwise paths, with explicit exceptions for hazardous aliasing.
- Widening integer sum/prod, integer mean, and direct reduction `out=`.
- Optional SIMD for a specified subset of float32/float64 and integer operations.
- Scalar fallback for every existing operation and every unsupported SIMD path.
- A separate `numjvm-linalg` module with float32/float64 `matmul`: vector-vector, vector-matrix, matrix-vector, matrix-matrix, and broadcast batches.
- A pure-Java implementation for the entire supported `matmul` surface.
- An optional OpenBLAS provider for float32/float64 GEMM, initially supported on Linux x86-64 with a documented LP64 CBLAS ABI.
- Correctness, memory-lifecycle, packaging, and performance validation, plus release documentation.

### 3.2 Deferred; not release requirements

- General multithreading and deterministic parallel reductions.
- LAPACK and Java fallbacks for `solve`, `inv`, `det`, `cholesky`, `qr`, `eigh`, and `svd`.
- `dot`, `vecdot`, complex matrix multiplication, and complex-function completion.
- MKL, Accelerate, and additional native ABIs/platforms.
- SIMD acceleration for all narrow/unsigned types, float16, bfloat16, complex types, and transcendental functions.
- A new uninitialized native-result allocator, GPU execution, expression fusion, and language-specific frontends.

Deferred work is recorded in Section 12. It may inform extension points, but it must not quietly become part of the M2 exit criteria. Any addition requires an explicit scope and estimate revision.

## 4. Proposed design decisions

### 4.1 Module and backend boundaries

Keep scalar execution in `numjvm-core`. Add an optional `numjvm-vector` module for Vector API implementations and `numjvm-linalg` for multiplication. The core must remain loadable without the Vector module and without native libraries. Do not expose incubator types in public core APIs.

Use a narrow internal provider boundary. Keep dtype resolution independent of backend choice: an accelerator implements an already-resolved operation, rather than changing promotion rules. The exact JPMS/service arrangement is settled in A04 and must work on both classpath and module path.

Proposed configuration:

| Setting | Behavior |
| --- | --- |
| SIMD `auto` | Default: use the optional provider when installed and resolved; otherwise use scalar execution. |
| SIMD `off` | Never select SIMD, including when the provider is installed. |
| SIMD `required` | Fail initialization clearly if the provider is unavailable; unsupported individual operations still use documented scalar paths. |
| Linear algebra `auto` | Use a compatible OpenBLAS provider when configured/discovered and beneficial; otherwise Java. |
| Linear algebra `java` | Force Java multiplication. |
| Linear algebra `openblas` | Require a compatible native provider; fail clearly if unavailable. |

Names are proposals to be finalized in A05. Read configuration once per execution context or initialization boundary; avoid a mutable global parallelism API in M2. Diagnostics should expose provider availability, selected provider, and fallback reason for tests/debugging without logging every numerical call.

### 4.2 Memory and temporary arrays

Preserve current factory initialization guarantees, including the existing behavior of `empty()`. Standard arena allocation does not provide a switch to disable zero-initialization. Profile before changing allocation policy.

Use bounded, leased scratch storage for eligible conversions. A lease cannot escape into a returned array. Nested/reentrant calls cannot reuse a live lease. Document its byte cap, cleanup policy, and relationship to scope ownership. Persistent scratch must be visible in memory accounting.

An 8,192-element chunk is a starting experiment, not a universal constant: use a byte cap so dtype and operand count cannot multiply memory use without bound.

For non-overlapping operations, conversion scratch should be bounded independently of input length. A hazardous overlapping input/output may require a full snapshot. Preserve correctness in that case, document the exception, and distinguish alias-protection allocation from conversion allocation.

### 4.3 Numerical contracts

Separate three contracts:

1. **Existing public semantics:** preserve promotion, overflow, broadcasting, layout, exceptions, and the current operation-specific NumPy oracle tolerances.
2. **Scalar/SIMD parity:** preserve finite result bits and signed-zero behavior for the accelerated subset. NaN classification must match; payload preservation is required only where explicitly specified. Existing tolerant oracle checks are not a substitute for these parity checks.
3. **New matrix multiplication:** Java and BLAS results use documented numerical tolerances, not cross-backend bit identity. The Java reference defines an explicit sequential multiply/add order for tests; optimized implementations are checked against the new matrix contract.

Avoid introducing FMA into existing elementwise kernels when it changes their result contract. Preserve the logical reduction tree for accelerated floating sums, independent of physical vector width. If a path cannot meet this, retain its scalar implementation.

For finite matrix tests, specify bounds using dtype epsilon, contraction length, and a scale such as `abs(A) @ abs(B)`, with an absolute floor near zero. Use a suitable higher-precision reference on small difficult cases and NumPy differential results on broader cases. NaN, infinity, overflow, and signed-zero cases need separate expectations, not a single relative tolerance.

### 4.4 Matrix multiplication contract

- Accept float32 and float64 arrays. Mixed float32/float64 operands produce float64 results; unsupported dtypes raise a clear dtype error in this release.
- Reject rank-zero inputs. Apply vector promotion/removal rules and broadcast leading batch dimensions.
- Support zero-sized dimensions, including a zero contraction dimension producing correctly initialized outputs.
- Provide a documented `out=` variant requiring the exact result shape, a writable output, and the exact result dtype in M2. Check overlap before execution.
- Support arbitrary valid input strides through direct access or packing. Use an internal contiguous result and copy back when output layout or aliasing requires it.
- Allocate results using a documented policy consistent with existing operand-scope behavior; scratch storage never owns an escaping result.
- Use Java for every supported shape/layout/dtype when no native backend is available. No factorization APIs are advertised in M2.

Batching uses repeated Java kernels or GEMM calls, without a NumJVM worker pool. A BLAS library may use its own threads; record and control those separately in benchmarks. Do not silently change process-wide BLAS thread configuration from a numerical call.

### 4.5 Native provider contract

Support one explicitly described ABI first: OpenBLAS CBLAS with 32-bit integer dimensions. Reject incompatible configuration or dimensions before a downcall. Do not claim reliable ABI detection from a generic library filename alone.

Implement native loading, symbol resolution, library lifetime, native-access guidance, layout adaptation, leading dimensions, and error translation as separate concerns. Heap-backed arrays require an appropriate native packing path. Native pointers remain valid for the complete synchronous call.

Fallback is allowed for unavailable providers or unsupported layouts/sizes before execution. Do not catch arbitrary native-execution failures and silently rerun after a partially written output.

## 5. Task conventions

Each row below is a task suitable for an issue or small reviewable change. Estimates are **ideal engineering days**, including local feature tests, code review response, and API documentation for the change. They exclude interruptions and contingency. A day is effort, not elapsed calendar time.

Dependencies identify required predecessors. “All A–D” means every implementation task in those workstreams. Every task must pass the common definition of done in Section 9 in addition to its row-specific criteria.

## 6. Implementation backlog

### A — Baselines, contracts, and backend infrastructure (6 tasks)

| ID | Deliverable | Acceptance criteria | Depends on | Days |
| --- | --- | --- | --- | ---: |
| A01 | Reconcile the implementation baseline and carry-over items | Record starting SHA, test/workflow state, current toolchain, existing ADR numbers, and release-follow-up status. Confirm whether Sonar baseline, required checks, and dependency maintenance remain open; file separate maintenance issues where not required for M2. | — | 0.5 |
| A02 | Establish the benchmark protocol and baseline runner | Reproduce representative M1 cases on designated hardware; save raw results, environment, warm-up/fork settings, backend/thread settings, and separate allocation/out/kernel measurements. Include small, medium, and large inputs. | A01 | 2 |
| A03 | Write numerical and memory contracts | ADR covers parity, reduction trees, matrix tolerances, supported operation/dtype matrix, alias exceptions, and scoped results. Add executable contract examples and reference-test helpers. | A01 | 1.5 |
| A04 | Implement the optional-provider/module boundary | Core starts without optional modules. Provider interfaces do not expose incubator types or unrestricted public segments. Smoke tests exercise classpath and module-path loading and absent providers. | A03 | 2 |
| A05 | Implement backend configuration and diagnostics | Auto/off-or-Java/required behavior is tested; invalid settings fail clearly. Tests can force a backend and verify that an accelerated kernel actually ran. | A04 | 1 |
| A06 | Establish backend CI jobs | Separate scalar-only and SIMD-enabled jobs build/run on the supported JDK; missing Vector module is tested. Define native CI setup and artifact retention without pretending unimplemented paths are validated. | A05 | 1 |

### B — Allocation, conversion, and reductions (8 tasks)

| ID | Deliverable | Acceptance criteria | Depends on | Days |
| --- | --- | --- | --- | ---: |
| B01 | Add allocation diagnostics and resolve allocator strategy | Measure heap/native/scoped allocation and native tracking separately; account for the 1 MiB threshold. Record evidence and choose safe M2 changes; do not introduce an uninitialized allocator by assumption. | A02, A03 | 1.5 |
| B02 | Implement bounded scratch leases | Byte cap, nested leases, cleanup, scope independence, and native accounting are tested. No returned array aliases scratch; no growth proportional to repeated operation count. | B01 | 2 |
| B03 | Implement buffered conversion iteration | Convert operands in bounded chunks while preserving broadcasting, strides, cast rules, and iteration order. Test both operands converting and tails smaller than a chunk. | B02 | 2 |
| B04 | Integrate conversion buffers with ufunc outputs | Eligible non-overlapping mixed-dtype ufuncs avoid full-input casts. Identical in-place views remain correct; hazardous overlap uses a documented snapshot. Test failures and lease release. | B03 | 1.5 |
| B05 | Implement widening sum/prod kernels | Bool and narrow signed/unsigned input accumulation matches existing result dtype and overflow rules without an input-sized widened array. Test axis lists, keepdims, empty inputs, and unusual strides. | A03 | 2 |
| B06 | Implement integer mean without a full conversion | Convert each input into the specified floating accumulator during reduction; preserve the reference accumulation order and final division/casting. Include large integers, unsigned values, cancellation, and empty reductions. | B05 | 1.5 |
| B07 | Implement direct reduction output execution | Compatible non-overlapping out arrays receive results directly. Validate output before work; preserve temporary fallback for aliasing or incompatible internal dtype. Cover all existing public reduction-out overloads. | B05, B06 | 2 |
| B08 | Optimize measured layout/allocation overhead | Address contiguous-axis reduction dispatch and output-order traversal where measurements justify changes; add a before/after report for allocation policy changes. Preserve layouts and initialization. A rejected optimization is documented rather than merged without benefit. | B01, B04, B07 | 1.5 |

### C — SIMD kernels (10 tasks)

| ID | Deliverable | Acceptance criteria | Depends on | Days |
| --- | --- | --- | --- | ---: |
| C01 | Implement the vector provider and generation scaffold | Optional module compiles generated kernels; capability selection, scalar fallback, bounded loads/stores, offset handling, and scalar tails are tested on heap/native segments. | A04, A05 | 2 |
| C02 | Generate float32/64 arithmetic kernels | Contiguous add/subtract/multiply/true-divide match the scalar contract for finite and exceptional values. Dtype resolution remains in core. | C01 | 1.5 |
| C03 | Implement scalar-broadcast arithmetic paths | Both left and right scalar operands, row broadcasting through iteration runs, tails, and compatible in-place outputs are correct; no broadcast materialization. | C02, B04 | 1.5 |
| C04 | Generate int32/64 arithmetic and bitwise kernels | Add/subtract/multiply and selected bitwise operations preserve overflow semantics. Integer true division remains on the correctly promoted floating path; unsupported operations fall back. | C01 | 1.5 |
| C05 | Generate comparison kernels and boolean stores | Float32/64 and int32/64 comparisons produce the correct boolean representation. Test NaNs, tails, nonzero offsets, and no writes beyond the output. | C01 | 1.5 |
| C06 | Generate floating extrema kernels | Maximum/minimum/fmax/fmin preserve their distinct NaN and tie/signed-zero rules; fallback remains available for unsupported dtypes and layouts. | C01 | 1.5 |
| C07 | Generate unary floating kernels | Float32/64 negate, absolute, square, and sqrt match the declared contract. Test negative zero, subnormals, infinities, and invalid sqrt inputs. | C01 | 1 |
| C08 | Implement vector pairwise floating sum | Preserve logical accumulators, 128-element blocks, recursive combination, and tails for float32/64. Test logical eight-accumulator mapping across supported vector widths and cancellation-heavy values. | C01, B07, A03 | 2.5 |
| C09 | Implement selected vector non-sum reductions | Add supported int32/64 sum and floating min/max paths, plus bool any/all paths where useful. Preserve identities, empty behavior, and scalar fallbacks; integer multiplication reduction remains scalar unless separately justified. | C04, C06, B07 | 2 |
| C10 | Add exhaustive kernel-path parity tests | Force direct vector invocation across supported species widths, offsets, run sizes, broadcast/tail paths, and negative-stride fallbacks. Assert dispatch coverage; unsupported hardware widths may use emulation for correctness, not performance claims. | C02–C09 | 2 |

### D — Portable matrix multiplication and OpenBLAS (12 tasks)

| ID | Deliverable | Acceptance criteria | Depends on | Days |
| --- | --- | --- | --- | ---: |
| D01 | Create the linalg module and API contract | Add module/build publication setup and experimental matmul signatures. Document dtype, out, ownership, and backend rules; core remains independently usable. | A03, A04 | 1 |
| D02 | Implement shape inference | Test vector promotion/removal, matrix cases, broadcast batches, zero dimensions, mismatches, and rank-zero rejection without allocating matrix-sized data. | D01 | 1.5 |
| D03 | Implement dtype, output, and lifetime validation | Float32/64 and mixed promotion, exact out dtype, read-only/closed/wrong-thread errors, size overflow, alias planning, and result allocator selection are tested. | D02 | 1.5 |
| D04 | Implement the Java reference executor | Simple tested contraction for vector and 2-D cases supports the full allowed real dtype set; define arithmetic order and zero-contraction behavior. It serves as a small-case reference and portable fallback foundation. | D03 | 1.5 |
| D05 | Implement blocked Java matrix kernels | Float32/64 contiguous kernels handle rectangular matrices and incomplete blocks; verify against the matrix contract and benchmark block sizes. | D04, A02 | 2.5 |
| D06 | Implement matrix packing and copyback | Handle C/F order, offsets, transposes, negative/zero input strides, heap/native storage, and strided outputs. Bound reusable panels where feasible; document full-result/alias temporary exceptions. | D03, B02 | 2 |
| D07 | Implement batched execution | Broadcast batch indexing composes with vector/matrix cases, Java kernels, and packing. Test empty batches and singleton dimensions; execute synchronously without a NumJVM worker pool. | D05, D06 | 1.5 |
| D08 | Implement OpenBLAS loading and ABI validation | Explicit path/configuration and controlled discovery; required symbols, supported LP64 ABI/configuration, native-access requirements, and library lifetime are verified. Missing/incompatible providers fail or fall back as configured. | D01, A05 | 2 |
| D09 | Implement and test the sgemm binding | Validate dimensions and leading dimensions before FFM calls; native-only tests verify float32 C/F/transposed rectangular cases, output buffers, and synchronous lifetimes. | D08, D06 | 1.5 |
| D10 | Implement and test the dgemm binding | Equivalent float64 binding with independent layout/argument tests; zero-sized cases handled without unsafe downcalls. No unsupported integer-width truncation. | D09 | 1 |
| D11 | Integrate native dispatch and fallback | Java and BLAS cover all supported shapes via direct calls or packing; small cases use measured thresholds. Test missing provider, dimensions outside native limits, forced backends, aliasing, and batch calls. | D07, D09, D10 | 2 |
| D12 | Extend the matrix oracle and property suite | NumPy shape/dtype/value cases plus finite error bounds, exceptional values, identity/transpose properties, and small higher-precision references pass on Java and BLAS. Check forced dispatch, not only returned values. | D11, A03 | 2 |

### E — Cross-feature verification and performance (8 tasks)

| ID | Deliverable | Acceptance criteria | Depends on | Days |
| --- | --- | --- | --- | ---: |
| E01 | Expand numerical stress fixtures | Add vector/block/chunk boundary lengths, cancellation, overflow, unsigned promotion, and exceptional values. Keep fixture size bounded and supplement with generated runtime tests. | B08, C10 | 1.5 |
| E02 | Validate output overlap across backends | Cross-feature tests combine mixed dtype, shifted/reversed views, identical in-place views, broadcast input, strided out, and matmul aliasing; no silent input corruption. | B08, C10, D12 | 1 |
| E03 | Validate scope and native-memory lifetimes | Automatic/confined/shared arrays, closed scopes, failed operations, scratch cleanup, repeated calls, and native library lifetime are tested. Establish bounded retained memory after warm-up and release behavior. | B08, D12 | 1.5 |
| E04 | Complete the execution-mode CI matrix | Full existing oracle passes in scalar-only and SIMD-enabled jobs. Linalg tests run Java-only and forced OpenBLAS; absent-provider startup and unsupported-path fallbacks are verified. | A06, C10, D12, E01–E03 | 1.5 |
| E05 | Validate distribution and external consumption | Separate consumer build exercises published core/vector/linalg artifacts on classpath and module path. Verify flags, dependency boundaries, source/Javadoc artifacts, and no mandatory native dependency. | E04 | 1 |
| E06 | Extend JMH and NumPy comparison suites | Measure allocating/out calls, mixed conversions, widening reductions, matrix shapes, packing costs, and backend choices. Native counters complement GC allocation profiling. | B08, C10, D12, A02 | 1.5 |
| E07 | Tune thresholds and run the performance campaign | Use the same designated hardware and numerical inputs for baseline/candidate. Publish three-fork raw JMH results, NumPy configuration, uncertainty, and regression decisions; tune dispatch without changing contracts. | E04, E06 | 2 |
| E08 | Audit the release gates | Check all required correctness, memory, packaging, quality, and performance evidence. Record unresolved defects or explicit scope changes; no waived correctness failures hidden in a benchmark report. | E05, E07 | 1 |

### F — Documentation and release (4 tasks)

| ID | Deliverable | Acceptance criteria | Depends on | Days |
| --- | --- | --- | --- | ---: |
| F01 | Publish architecture and ADR updates | Document actual providers, conversion buffers, numerical contracts, ownership, supported platforms, and deferred features. Allocate ADR numbers after A01; do not reuse an occupied number. | All A–D | 1 |
| F02 | Publish runnable examples and setup guide | Examples cover SIMD opt-in module setup, scalar fallback, Java matmul, OpenBLAS configuration, scoped results, and diagnostics. Execute examples in CI. | E05 | 1 |
| F03 | Publish benchmark report and retrospective | Include raw-data links, successes, misses, environmental limits, actual effort, and recommended follow-up sequence. Distinguish measured results from targets. | E07, E08 | 1 |
| F04 | Prepare and verify 0.3.0-alpha release | Changelog, version, consumer smoke test, release checklist, and roadmap agree with shipped behavior. Tag/publish according to maintainer policy after review of the release evidence. | E08, F01–F03 | 1 |

## 7. Estimates and delivery sequence

| Workstream | Tasks | Ideal days |
| --- | ---: | ---: |
| A — Baselines, contracts, and backend infrastructure | 6 | 8 |
| B — Allocation, conversion, and reductions | 8 | 14 |
| C — SIMD kernels | 10 | 17 |
| D — Portable matrix multiplication and OpenBLAS | 12 | 20 |
| E — Cross-feature verification and performance | 8 | 11 |
| F — Documentation and release | 4 | 4 |
| **Total** | **48** | **74** |

Allow **25–30% contingency**, giving approximately **93–96 engineering days**. At five fully available engineering days per week, that is roughly **19–20 working weeks for one developer**. At four effective days per week, allow approximately **23–24 weeks**. Holidays and unavailable contributors extend elapsed duration.

These estimates include focused feature testing. They are planning judgments, not measurements of future delivery speed. Re-estimate after A02–A04, after the first complete vector kernel, and after the first working native GEMM binding.

Do not infer that forty ideal engineering days imply forty elapsed calendar days from historical commit dates. Use actual available capacity. Parallel contributors may shorten elapsed duration, but provider contracts, shared dispatch changes, native integration, and review remain dependencies.

Recommended integration sequence:

1. **Baseline and contracts:** A01–A06; begin B01 only after the reference measurements and memory contract exist.
2. **Scalar performance improvements:** B01–B08. This creates a useful intermediate checkpoint even before SIMD or native integration.
3. **SIMD execution:** C01–C10. D01–D08 can proceed independently once their A/B prerequisites are complete, if contributors are available.
4. **Complete multiplication:** D09–D12, integrating the Java path before making native acceleration the default eligible choice.
5. **Integration, measurements, and delivery:** E01–E08 and F01–F04. Add tests/docs during development; these tasks consolidate release-level evidence.

No fixed dates are assigned until start date and contributor capacity are known. Milestone completion is determined by evidence, not by a preselected number of commits.

## 8. Release acceptance criteria

### 8.1 Mandatory correctness and portability gates

- Existing scalar behavior and oracle tests continue to pass. SIMD-enabled runs additionally demonstrate actual vector-path execution and scalar/SIMD parity for the supported subset.
- Unsupported accelerated paths fall back without changing promotion, overflow, shape, layout, or exceptions.
- All advertised matmul cases pass in a Java-only environment. Forced OpenBLAS runs pass the same public contract with the documented numerical tolerances.
- Core and Java linalg consumers work without Vector API resolution, Python, or a native numerical library at runtime.
- Output overlap, scopes, closed-memory access, empty arrays, offset views, and large-size validation pass the specified tests.
- The new module's public API does not advertise LAPACK functions or unsupported native providers.

### 8.2 Mandatory memory gates

- Eligible non-overlapping mixed-dtype operations use conversion scratch bounded independently of total input length after warm-up.
- Widening sum/prod and integer mean avoid an input-sized conversion array on their supported paths.
- Compatible direct reduction out paths avoid an intermediate result-sized array.
- Hazardous overlap and matrix packing/copyback exceptions are documented and tested; a blanket zero-temporary promise is not made.
- Native allocation counters and lifecycle tests accompany JMH GC profiling. Scratch reuse does not leak memory or return arrays referencing reusable scratch.

### 8.3 Performance protocol and targets

Benchmark comparisons must use the same machine, dtype, logical values, layouts, allocation mode, backend, and relevant thread settings. For BLAS comparisons, record the actual library and version used on both sides. Separate packing-inclusive timings from already compatible input timings.

Use at least three JMH forks and sufficient warm-up; retain raw samples and uncertainty. Report NumPy timing methodology separately. Shared CI machines run correctness and smoke benchmarks; performance release evidence comes from the designated reference environment, which should have stable access to resources.

| Case | Proposed target on the reference environment | Treatment |
| --- | --- | --- |
| Contiguous float64 add/multiply with preallocated output, about 1M elements | At most 1.5× NumPy time | Confirm achievable measurement protocol in A02; assess in E07. |
| Contiguous float64 sum, about 1M elements | At most 1.5× NumPy time | Exactness contract takes precedence; document a scalar fallback if required. |
| int32 sum, about 1M elements | At most 2× NumPy time | Validate both speed and removal of the widened temporary. |
| Allocating operations | Material improvement against M1 for targeted sizes without a material small-array regression | A02 records a case-specific budget; no promise that native allocation changes fix heap cases. |
| 1024×1024 float64 multiplication, same BLAS and thread count | At most 1.2× NumPy time | Report preallocated and allocating APIs separately, including packing where applicable. |
| Java multiplication | Correct portable execution, plus published throughput across representative sizes | No universal “within 10× BLAS” release guarantee before a Java baseline exists. |
| Layout sensitivity | Preserve the applicable M1 transposed-sum and broadcast-add targets | Use equivalent workloads and report incompatible-layout exceptions explicitly. |

The numerical targets are initial performance objectives. A02 must establish the reference environment and record the final measurable performance gate before optimization. E08 either confirms the gate or records an explicit maintainer-approved target/scope revision with evidence. Do not silently relabel a missed target as achieved. Correctness, memory safety, and fallback gates cannot be relaxed to meet speed goals.

There is no multiworker scaling gate in M2 because NumJVM parallel execution is deferred.

### 8.4 Quality and delivery gates

- Build, oracle, backend integration, and agreed static-analysis checks pass on the release commit.
- Configure branch-required checks where repository policy and permissions allow; record configuration separately from source-code changes.
- Target at least 80% line coverage in each library module, measured consistently and with exclusions documented. Coverage complements targeted boundary and failure tests; it does not replace them.
- Generated/vector/native integration paths have direct tests even where source coverage attribution is imperfect.
- Documentation, examples, changelog, benchmark report, and consumer smoke tests describe the actual release surface.

## 9. Common definition of done

Every implementation task is complete when its deliverable and row-specific checks are reviewed, applicable unit/property/oracle tests pass, public behavior is documented, and no unexplained correctness or memory regression remains. Tests must verify behavior and failure modes, not merely repeat implementation details.

For performance tasks, attach before/after measurements with comparable inputs and settings. For backend tasks, demonstrate the requested backend actually executed. For memory tasks, include both heap allocation evidence and relevant native/scratch accounting.

Issues should include: task ID and title, problem statement, implementation boundary, dependencies, acceptance criteria, validation commands/results, effort estimate, and known risks. Link follow-up issues for excluded work rather than adding it silently.

## 10. Principal risks and responses

| Risk | Response and decision point |
| --- | --- |
| SIMD changes loading requirements of core | Prove absent-provider startup on classpath and module path in A04 before adding kernels. |
| Scratch reuse violates scope ownership or alias safety | Leased internal buffers, bounded retention, snapshot exceptions, and E02/E03 stress tests. |
| Acceleration changes floating reduction results | Fix the logical arithmetic tree in A03/C08; keep scalar paths when parity cannot be achieved. |
| Allocation costs have a different cause from the report's hypothesis | B01 profiles the real path; reject unjustified allocator changes. |
| Native ABI assumptions cause unsafe calls | Start with one specified ABI; validate arguments and supported library configuration before dispatch. |
| Packing overwhelms small-matrix acceleration | Benchmark end-to-end costs; choose Java below measured thresholds and expose backend diagnostics. |
| Exactness/performance goals conflict | Preserve correctness and publish the tradeoff; revise a performance target only explicitly. |
| Native testing expands across platforms | Linux x86-64 OpenBLAS is the supported native baseline; other providers stay deferred. |
| Maintenance changes destabilize the release | Audit in A01; apply necessary blockers as separately estimated work instead of blindly upgrading all dependencies. |
| Scope grows into a general linalg library | Keep only matmul mandatory; use the follow-up sequence below. |

## 11. Changes relative to the supplied proposal

The supplied document has 36 tasks and 41.5 ideal days. This proposal makes the following changes:

- Splits broad backend, conversion, SIMD, and matrix work into independently verifiable tasks.
- Replaces assumed uninitialized native allocation with evidence-driven profiling and a bounded scratch strategy.
- Explicitly includes integer mean and direct reduction out improvements.
- Limits initial SIMD and native dtype/provider coverage.
- Makes the Java fallback promise complete and implementable for matmul, rather than implicitly promising Java implementations of all LAPACK operations.
- Defers general threading, factorizations, and complex expansion, and removes their release gates.
- Separates public semantic compatibility, scalar/SIMD parity, and matrix numerical tolerances.
- Adds explicit alias, lifetime, backend-dispatch, native-memory, and consumer-package validation.
- Separates engineering effort from elapsed calendar duration and includes contingency.

Despite the narrower feature scope, more detailed tasks and explicit integration work can increase the effort estimate. Task count and estimated time are not proportional measures of functionality.

## 12. Follow-up roadmap

These are future planning topics, not additional tasks counted in the 48-task M2 backlog. Confirm version numbering after M2; a substantial new API may merit a new minor alpha release rather than a patch release.

| Follow-up | Intended content | Prerequisite decisions |
| --- | --- | --- |
| Parallel CPU execution | Opt-in elementwise parallelism first; deterministic reductions afterward | Worker pool lifecycle, all-operand accessibility, fixed arithmetic tree, aliasing, error propagation, scratch budgeting, BLAS oversubscription. |
| Linear algebra foundations | LU-based solve and determinant, then Cholesky; inverse only with a clear use case | Native-only versus Java/dependency fallback, singularity errors, workspace management, dtype and batching coverage. |
| Advanced factorizations | QR, symmetric/Hermitian eigensolvers, SVD | Reconstruction/residual/orthogonality tests; eigenvector/sign/subspace ambiguity; convergence failures; fallback and platform support. |
| Numerical compatibility expansion | dot/vecdot, complex arithmetic gaps, complex multiplication, additional SIMD dtypes | Distinct shape and conjugation semantics, branch-cut/overflow behavior, operation-specific accuracy contracts. |
| Additional native providers | MKL, Accelerate, more OS/architecture combinations | ABI mapping, licensing/setup documentation, automated native test environments. |
| Existing longer-term roadmap | Fusion/JIT, language APIs and data interchange | Revisit ordering after measured M2 results; preserve the existing array model where possible. |

## 13. References

Repository references below are pinned to the analyzed commit. They describe the review baseline, not an assertion that the branch cannot change before implementation.

- [Roadmap](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/ROADMAP.md)
- [Architecture](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/ARCHITECTURE.md)
- [M1 benchmark results](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/docs/benchmarks/M1.md)
- [Operand conversions](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-core/src/main/java/org/numjvm/ufunc/Operands.java)
- [Binary ufuncs and reductions](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-core/src/main/java/org/numjvm/ufunc/BinaryUfunc.java)
- [Reduction implementations](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-core/src/main/java/org/numjvm/ufunc/Reductions.java)
- [Native allocator](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-core/src/main/java/org/numjvm/internal/memory/NativeAllocator.java)
- [Pairwise summation](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-core/src/main/java/org/numjvm/internal/loops/Pairwise.java)
- [Scopes](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-core/src/main/java/org/numjvm/memory/NJScope.java)
- [NumPy oracle contract](https://github.com/stamich/NumJVM/blob/2aeff90a1d68f405ed6b635c7de929015ba54af9/numjvm-tests-oracle/README.md)
- [Java 25 Arena API: initialization and lifetime guarantees](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/foreign/Arena.html)
- [NumPy matmul semantics](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- [NumPy dot semantics](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
- [NumPy vecdot semantics](https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html)
- User-supplied source: *NumJVM 0.3.0 — milestone proposal.md*, dated 29 October 2026; reviewed in this conversation.
