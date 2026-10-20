# Numerical specification checks

These scripts verify small mathematical claims; they are not model implementations or empirical evaluations.

| File | Status and purpose |
|---|---|
| `kernel_check2.py` | Canonical check cited by the current specification. Compares the fixed-centre causal support 0..60 with matched support -30..30, and approximates a wide two-sided reference on -300..300. |
| `kernel_check2.out.txt` | Recorded canonical output. The four tested widths satisfy the lag-12 bound; width 18 refutes a universal all-lag 1e-4 claim. |
| `kernel_check.py` | Historical exploratory check, retained for provenance. Uses a reference on -200..200 and an older learned-centre, 0..48 construction. It is not the current kernel specification. |
| `kernel_check.out.txt` | Recorded output of that historical experiment; not evidence for the current fixed-centre kernel's truncation error. |

The two reference supports are both finite approximations to an infinite window, and the scripts also differ in the causal kernel being tested. They should not be compared as duplicate implementations of one check. Only `kernel_check2.py` supports the current manuscript-change record. Do not edit historical output to make it resemble the canonical experiment.
