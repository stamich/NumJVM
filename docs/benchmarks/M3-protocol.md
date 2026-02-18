# M3 expression performance protocol

`ExpressionBenchmark` compares a float64 multiply/add/divide chain in eager,
block and generated modes. It includes separate allocating and preallocated
variants. Run with scalar eager kernels and with `-PsimdBench` plus the required
Vector API settings, recording which provider actually ran.

Use a designated reference machine and at least three forks, recording raw JMH
results, uncertainty, Java version, CPU, memory, backend and thread settings.
Measure graph construction and first-call compilation separately from warm
execution. Add representative mixed-dtype, strided, broadcast and in-place
cases before claiming the full M3 performance gate.

No reference machine has been designated here; the M2 and M3 performance gates
remain open. CI and shared-machine runs serve as infrastructure checks only.
