# Stage 7 controlled simulation design

Recorded 2026-10-21 before simulation fits. This supplements the implementation
plan; it changes no frozen probability law, historical split or scoring rule.
Only generated data enter this stage. No research panel or held-out outcomes
are read. Stage 8 remains responsible for production model variants and
development selection.

## Questions and execution

1. Verify NB2, censored marks, masking, stationary factors, triangular loadings,
   kernel convolution and HMM marginalization against independent calculations.
   Compare the HMM with exhaustive enumeration through T=6, R=3; check emission
   and transition gradients, missing months, smoothing and backward sampling.
2. Generate controlled datasets with independent counts, AR latent counts,
   shared factors and sparse categories, severity and missing auxiliary channels,
   regime transitions, reporting/threat changes, and present/absent kernels.
   The generator is NumPy/SciPy code separate from the PyMC likelihood.
3. Run four-chain NUTS on three fixed seeds (7101, 7102, 7103) per scenario.
   Use the frozen 1000/1000, 2000/1000, 2000/2000 warmup/retained budgets,
   target acceptance .95/.99/.99 and depth 12/15/15. Retain failed attempts.
   Require the existing R-hat, ESS, BFMI, divergence and depth gates; do not
   convert a failed fit into recovery evidence.
4. Report latent and observable 50%/90% interval coverage, RMSE, parameter
   intervals and posterior contraction. These fixed-truth stress cases are
   **not simulation-based calibration (SBC)**. Three datasets do not establish
   repeated-sampling calibration; cells within a dataset are dependent.
   A separately labelled known-parameter predictive calibration experiment
   checks the observation/forecast simulator, not posterior calibration.
5. Refit the count-derived local-linear-trend effort preprocessing separately
   on every generated training dataset. Reporting-only and threat-only shifts
   with the same sum in log mean have the same count likelihood. Compare the
   proxy with no adjustment and an oracle offset; distinguish the operational
   reference intensity from the generating physical interpretation.
6. Test tau/psi separation in additional conditional experiments at dense and
   sparse rates, with the baseline level fixed at truth to give separation its
   best chance. If separate recovery is weak, follow SPECIFICATION section 3.4:
   omit iid latent count noise from the initial factor pilot and retain NB
   dispersion. Keep the full residual experiment as evidence, not a silent
   change. Factor fixtures without iid noise are conditional submodels until
   this decision is evaluated.
7. Compare kernel-present and kernel-absent fixtures with shared AR factors.
   Assess width and loading norm contraction, covariance recovery and absence
   shrinkage. Failure identifies an unsupported extension, not a reason to
   choose another seed. Assess truncation on widths .25 to 18 and lags 0 to 120
   against a two-sided reference on -300..300; report maxima, not a uniform
   mathematical error bound.

The small recovery fixtures use four categories (last sparse) and 96 months;
kernel fixtures use 144 months. The standalone AR and noise-separation cases
use one category. A representative count-factor timing fixture uses ten
categories, two factors and 108 months. Record the exact switches and generating
parameters with each dataset. Synthetic category order is declared in the
fixture, with dense anchors first; it does not reorder the real panel.

Auxiliary-channel component checks use known synthetic exposure and calibration
coefficients. They do not establish that real EPSS/KEV/SEC channels are eligible
or that their coefficients can be recovered from the available real record.
Misspecified observations and missingness are deliberate stress experiments;
failure of nominal coverage there is informative, not a software test failure.

## Evidence and interpretation

Save data/truth hashes, source hashes, configuration, all posterior attempts,
diagnostics and recovery summaries. Separate graph construction, compilation,
first draw/startup, warmup, retained sampling, diagnostics and serialization
where directly observable; do not claim a callback isolates operating-system
worker startup. Record per-chain wall markers and runtime memory observations.
Small synthetic timing is evidence for feasibility, not the Stage 9 rolling-fit
budget. Preserve original freezes and Stage 6 results.

Recovery is judged on eta, expected observed counts, severity locations and
implied covariance, not arbitrary factor rotations. Ordered regime shifts fix
labels, but within-fit filtering remains retrospective through continuous
parameter/state estimation. Weak identification and failed computational gates
must be carried forward explicitly. No minimum empirical coverage from three
seeds is used as a newly invented statistical acceptance threshold.

Method references: [Stan's SBC description](https://mc-stan.org/docs/2_37/stan-users-guide/simulation-based-calibration.html)
distinguishes prior-generated calibration from fixed-truth recovery;
[PyMC's HMM example](https://www.pymc.io/projects/examples/en/latest/mixture_models/discrete_markov_chain_marginalization.html)
describes forward marginalization followed by state recovery. This project
implements and independently checks its own recursion without adding a new
dependency or changing the lock file.
