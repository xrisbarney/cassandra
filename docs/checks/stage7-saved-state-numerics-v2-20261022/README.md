# Saved-state checks after the cancellation diagnosis

The [original review](../stage7-saved-state-numerics-20261022/README.md) preserves
the single-step failure and its independent 60-digit resolution. Its original
checker is retained in Git commit `656e148`.

The follow-up uses the unchanged native/Numba density and gradient tolerances.
For selected first/last coordinates of every parameter block, central differences
are evaluated at the fixed steps 0.003, 0.001, 0.0003, 0.0001, 0.00003 and 0.00001.
Two adjacent estimates must agree with each other and the analytic gradient
within the original 5e-5 absolute/relative tolerance. All estimates are saved,
including the small-step failures. This changes the numerical diagnostic step
choice after an independently resolved cancellation finding; it changes no
sampling, recovery or statistical acceptance rule.

Both checkpoints pass: primary fallback seed 7102 and wider-dynamics fallback
seed 7101. Maximum native/compiled gradient differences are respectively
1.10e-7 and 2.89e-8. The representative checkpoint passed the original single-step
check. Saved states are not necessarily the rejected leapfrog proposals; these
local checks neither waive the divergences nor establish global convergence.

The [audit](audit.json) binds source and output hashes. No production model,
prior, generated observation or held-out data was changed.
