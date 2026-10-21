# Bound native numerical threads per fit process

2026-10-21. Inspection found NumPy OpenBLAS configured for eight threads on an
eight-logical-CPU host. Stage 7 coordinates three fit processes, each with four
sequential chains. Starting with this patch, each newly launched fit sets
OpenBLAS/OpenMP/MKL/Numba thread environment variables to one and limits loaded
native pools to one using threadpoolctl. Record the actual pools in run.json.

This changes execution resources, not the probability law, sampler budgets,
diagnostic gates, or selected datasets. Floating-point reductions may differ;
do not claim bitwise posterior identity. Existing processes finish with their
recorded code/settings. Separate timing summaries by source/thread policy; do
not retroactively treat older measurements as single-thread measurements or
rerun accepted fits merely to improve timings.

[threadpoolctl documentation](https://github.com/joblib/threadpoolctl) describes
controlling native pools to avoid nested parallelism and oversubscription.
