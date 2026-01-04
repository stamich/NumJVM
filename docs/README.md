# NumJVM documentation

| Document                                         | Contents                                                        |
|--------------------------------------------------|-----------------------------------------------------------------|
| [Architecture](../ARCHITECTURE.md)               | Modules, packages, key decisions, where later milestones attach |
| [Memory model](design/memory-model.md)           | Ownership, scopes, heap vs native, threads, failure modes       |
| [NDArray model](design/ndarray-model.md)         | Metadata, invariants, element access, views, reshape, iteration |
| [Dtype system](design/dtype-system.md)           | The 15 dtypes, conversions, uint64, NumPy descriptors           |
| [Broadcasting](design/broadcasting.md)           | Rules, broadcast views, use in kernels                          |
| [ADRs](adr/README.md)                            | Allocation model, element access API, iterator shape            |
| [M0 benchmarks](benchmarks/M0.md)                | JMH results, NumPy comparison, reproducibility                  |
| [M1 benchmarks](benchmarks/M1.md)                | Ufuncs and reductions by layout and dtype, against NumPy        |
| [NumPy oracle](../numjvm-tests-oracle/README.md) | Differential tests against NumPy                                |
| [Code quality](quality/README.md)                | SonarCloud scope, CI analysis, quality gate                     |
| [M0 summary and retrospective](milestones/M0.md) | Exit criteria, plan vs actual, lessons, M1 inputs               |
| [Roadmap](../ROADMAP.md)                         | Milestones M-1 to M6, re-estimated M1 plan                      |
