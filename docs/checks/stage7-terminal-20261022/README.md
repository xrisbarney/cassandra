# Saved-state audit before the final diagnostic round

Both preserved callback states and two fixed 1e-4-SD perturbations per state
passed complete native/compiled density and gradient comparisons, independent
dense SciPy density reconstruction, and independent finite differences of the
first/last coordinate of each value variable. The tolerance records and actual
comparisons are in `masked_fallback.json` and `matched_one_factor_fallback.json`.
No posterior was sampled. These local checks do not certify entire NUTS
trajectories; the callback point is not the full divergent trajectory.

`executed_audit_source.py.txt` preserves the exact audit script bytes hashed in
the records. The maintained script only removes an unused import afterward.
`scripts/stage7_terminal_oracle.py` uses dense SciPy Cholesky/solves, normalized
SciPy probability laws and explicit transform Jacobians; it does not call the
custom block Op or its derivative. It checks the statistical target as well as
agreement between execution backends.

At the matched one-factor callback point, phi=0.96517934 corresponds to an
e-folding correlation time of about 28.2 months. At the masked two-factor point,
the second phi=0.99948395 corresponds to about 1,937 months. Thus a rule that
merely makes memory shorter than 108 months does not automatically exclude the
one-factor failure neighborhood. Neither observation identifies the exact
location where the divergent trajectory accumulated integration error.

The [saved scale calculation](saved_scales.json) gives m_sd=0.00334668 in the
matched one-factor point and 1.30954 in the masked two-factor point. The former
is a narrow hierarchical region; its shift-prior curvature is about 573,240,
not close to flat in that direction. This strengthens the reason to test joint
level/state coordinates before attributing the problem solely to persistence
or changing its prior. It does not establish the complete divergence mechanism.

There is an exact **likelihood** invariance for a fixed loading column gamma:
shift all states of that factor by c and category levels by -gamma*c. Under the
current stationary unit-variance AR prior and proper Gaussian level covariance
C, the negative-log-prior curvature in c is

    gamma' C^-1 gamma + 1 + T (1-phi)/(1+phi),

which is strictly positive for |phi|<1. The unit term comes from the initial
stationary factor state. Consequently the likelihood tradeoff need not be an
exactly flat posterior direction. The reported curvatures at the saved points
are diagnostic calculations, not posterior summaries. Joint coordinates can
improve conditioning but cannot create missing empirical information.

The [one-factor joint verification](joint_verification.json) passed at full size:
three free-persistence points (initial, perturbed, near-unit) and two fixed-
persistence points. It checks dense reconstruction, complete native/compiled
targets and gradients, independent finite differences and correspondence with
the original conditional-state coordinates. The exact executed verifier is
preserved in `executed_verification_source.py.txt`; the maintained verifier only
wraps one long output line afterward. Neither source cleanup changes equations.
The prior and original Stage 7 implementation remain unchanged.
The [terminal plan](../../STAGE7_TERMINAL_DIAGNOSTIC_PLAN.md) fixes the global
deadline, five-attempt maximum and the final not-admitted outcome on failure.
