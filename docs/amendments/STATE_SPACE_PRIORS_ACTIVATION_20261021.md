# Activation of the Stage 7 stationary-amplitude prior

2026-10-21. Authority: the user instructed, "continue stage 7. patch issues as
you find them", after reviewing the completed prior amendment draft. Activate
the exact laws in `STATE_SPACE_PRIORS_20261021.md` as `stationary_bounded_v1`.
The sealed draft and original Stage 2 freeze are retained unchanged. The
companion JSON records the specification hashes before revised posterior fits.

This activation concerns synthetic Stage 7 fixtures and their forecast/recovery
code. It does not activate real-data fitting or held-out scoring. The broad
prior-predictive tails remain a disclosed limitation requiring plausibility and
sensitivity checks before production selection; finite moments are not a claim
of empirical calibration. No posterior performance score chose the new bounds.

Use new directories under `artifacts/stage7/recovery-v4`. Reuse the original
40 scenario/seed combinations and generated observations. Preserve original
truth coordinates alongside the translated stationary-amplitude truth. Do not
reuse old-prior posterior draws as amended-prior validation. The conditional
auxiliary fixture retains its known coefficients: it does not fit the new KEV
slope prior or establish real-channel identification.

Before sampling, verify transition densities and native/NUMBA derivatives,
truncated-prior normalizers, paired truth equivalence, and forecast moments.
Run independent/latent-AR/sparse-factor fixtures first, then the remaining
fixtures. Retain the existing four-chain budgets, retry schedule and every
diagnostic gate. Keep unsuccessful fits and report failures with denominators.
The already documented fail-fast rule applies. No time limit, shorter retained
budget, new convergence tolerance or score-based retry is introduced.

The existing count-derived effort experiments and nonidentification proof
remain part of Stage 7. Prior-generated SBC is a separate gate before any
general posterior-calibration claim; the fixed-truth recovery campaign itself
must continue to be described as controlled stress testing.
