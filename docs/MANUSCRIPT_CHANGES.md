# Stage 1: Record of changes required in the manuscript

The dated Stage 7 addendum below supersedes the original Stage 1 recommendations
about factor innovation covariance, loading priors and the residual noise floor.
The historical Stage 2 specification remains sealed; implementation amendments
are recorded separately.

Each entry names the location in `main.tex`, what it currently says, what the
specification (`SPECIFICATION.md`) proposes, and the edit needed once the
corresponding implementation and evidence exist. The numbered table records the
Stage 1 design review; later implementation evidence is appended below.
Factual and mathematical review updated on 2026-10-20. Entries are ordered by section.
"Pending evidence" entries are claims the specification cannot support until
the named stage produces results.

The user supplied a failed evaluation solely as a source of comparator identities
and measurement families. The [carry-forward record](PRIOR_EVALUATION_REVIEW.md)
and [evaluation protocol](EVALUATION_PROTOCOL.md) implement that scope. No old
values, rankings, or conclusions are carried into the new results. January
2024–December 2025 is the held-out forecast period for the rerun, with design
and tuning restricted to data through December 2023. Describe that separation
accurately without claiming the period has never been examined in earlier work.

| # | Location | Manuscript now | Specification | Required edit |
|---|---|---|---|---|
| 1 | §3 Problem Formulation, definition of $\lambda_{k,t}$ | "the true, unobserved level of malicious activity and exploitable exposure" | expected publication rate at a reference effort-proxy value (SPEC §3.1) | Define the operational rate and origin-dependent reference. Predictive association with other channels is evidence of usefulness, not proof that it measures malicious activity or a causal effect. |
| 2 | §3, exploitation signal $E_{k,t} \in [0,1]$ "mean or upper-quantile EPSS score or CISA KEV inclusion counts" | one channel with two incompatible supports | two channels: $E$ (mean EPSS, probability) and $V$ (KEV additions, count) (SPEC §1, §5.4) | Introduce $V_{k,t} \in \mathbb{Z}_{\ge 0}$ as a separate channel with its own NegBin likelihood, or state that KEV is not used in the primary model. Update $\mathcal{F}_t$, Figure 1 and Table 1 ("Multi-channel") accordingly. |
| 3 | §3, $\mathcal{F}_t$ includes $L_{s,\tau}$; §4 "SEC 8-K incident filings $D_{s,t}, L_{s,t}$" treated as observations of the latent state | loss marks are listed as an observation channel | $L$ is not a likelihood term; reserved for damage calibration and evaluation (SPEC §6) | Remove $L$ from $\mathcal{F}_t$ and from the list of measurement channels. Describe loss marks under Economic translation as calibration and evaluation inputs. |
| 4 | §4 opening: "a single Bayesian hierarchical state-space model" | one joint posterior including economic translation | modular: technical posterior fitted first, economic layer calibrated conditionally on its draws (cut model) (SPEC §6) | Rephrase as a two-module framework: a Bayesian state-space model of the threat landscape, and an economic layer calibrated conditionally on its posterior. State explicitly that loss evidence does not feed back into the technical posterior. |
| 5 | §4.2, Eq. factordyn: $\bm{w}_t \sim \mathcal{N}(\bm{0}, \bm{Q})$ with free $\bm{Q}$; identification text says the triangular constraint on $\bm{\Gamma}$ alone picks one representative | scale of each factor remains free when $\bm{Q}$ is free | $\bm{Q} = \bm{I}$; loadings carry the scale (SPEC §3.3) | Replace $\bm{Q}$ by $\bm{I}$ in Eq. factordyn and in the forecast recursion. In the identification paragraph add: "the factor innovation covariance is fixed to the identity, which together with the triangular constraint removes rotation, scale and sign freedom." Remove $\bm{Q}$ from the list of static parameters in Figure 1 and §4.5. |
| 6 | §4.2: $\bm{\Phi}$ written as a general matrix | general VAR(1) factors | diagonal $\bm{\Phi}$ (independent AR(1) factors) as default (SPEC §3.3) | State that $\bm{\Phi}$ is diagonal with entries in $(-1,1)$, or report the full-$\bm{\Phi}$ variant if Stage 8 adopts it. |
| 7 | §4.2: regime levels $\bm{\mu}_{z_t}$, $\bm{\nu}_{z_t}$ with no label-identification statement | regime labels unidentified | common ordered shift $\mu_{r,k} = m_k + c_r$, $c_1 = 0 < c_2 < \dots$; $\nu_{r,k} = n_k + d_r$ (SPEC §3.2) | Add the parametrisation and the ordering constraint; state that regimes are ordered by the common intensity shift. |
| 8 | §4.2: "$z_t$ ... shifts the levels"; initial regime distribution unstated | | $z_1$ uniform (SPEC §3.2) | State the initial distribution. |
| 9 | §4.2, Eq. kernel and covariance interpretation | sum support and normalisation unspecified; sigmoid tails are nonzero; width collides with horizon $h$ | finite causal kernel with fixed centre $c_0$, width $b$, squared-weight normalisation, and scale in $\bm a$ (SPEC §3.5) | State equality in distribution only with the correspondingly truncated and normalised two-sided process. Report truncation checks with their tested widths/lags, not a universal $10^{-4}$ bound. The width controls shape, while the finite support sets the exact covariance cutoff. |
| 10 | §4.2: "shrinks to zero in the posterior if the data do not support a distinct local dependence beyond $\bm{\Phi}$" | asserted | pending evidence, Stage 7 synthetic design | Keep only if the Stage 7 recovery experiment shows it; otherwise report the kernel as weakly identified against the AR factors. |
| 11 | §4.3: standardised smooth trend of aggregate log CVE volume | estimator and interpretation underspecified | origin-conditional filtered trend treated as a fixed proxy (SPEC §2) | State the window, missing-data handling, standardisation, and omitted proxy-estimation uncertainty. The adjustment can absorb genuine growth but does not force all growth into effort. A smoother fitted only through the origin does not use data beyond that origin; historical within-fit values can still be retrospective. |
| 12 | §4.3, Eq. obs-cvss and severity dynamics | unbounded lognormal channel for bounded CVSS | coarsened logit-normal default; no separate iid latent severity noise; lognormal family as an alternative (SPEC §3.4, §5.3) | Specify interior densities and endpoint tail masses, and the common scoring scale. Define default severity as a normalized CVSS location/median, not a probability or mean. Distinguish revised-data and reconstructed as-of score vintages. Recalibrate economics if the severity family changes. |
| 13 | §4.3, Eq. obs-epss: $E_{k,t}\in[0,1]$ | cohort, vintage, and boundary likelihood unspecified | scored publication cohort in an acceptable snapshot; coarsened probability target with endpoint masses and snapshot-version steps (SPEC §1, §5.2) | State the cutoff, maximum snapshot age, available-score count, and induced censored likelihood. Define version indicators from the actual file, and distinguish measurement changes known at the origin from future changes. |
| 14 | §4.3, Eq. obs-8k: Poisson incidents | changing reporting population and target unit unstated | incident–registrant pairs; step from July 2024, with December 2023 and June 2024 masked (SPEC §1, §5.5) | State the legal dates and full-month exclusions. A registrant count multiplies the rate; its logarithm is an additive log-rate offset. Advance announced population changes in forecasts. Preserve separate registrants affected by one shared event. |
| 14a | §3 and §4: KEV inclusion counts | launch backlog and entry-date interpretation unstated | November 2021 excluded; later catalogue additions are observations, not dates of first exploitation (SPEC §1) | State the launch-backlog exclusion. Avoid an exact launch count without a dated catalogue snapshot; CISA's fact sheet gives approximate components. |
| 15 | §4.3: "the four channels are asserted to be conditionally independent, so each source sharpens and cross-validates the latent state rather than double-counting it" | the second clause is a consequence claimed, not shown; $E$, $B$ (and $V$) are computed from the same CVE records as $N$ | tested in Stage 14 (SPEC §5.6) | Keep the assumption; drop "rather than double-counting it" until the residual-dependence test is reported. |
| 16 | §4.4: damage function described in words only | | exact zero-anchored sigmoid with sector-relative operating point (SPEC §6) | Write the equation for $\phi_s$ with its three parameters and the definition of $\bar S_s$. |
| 17 | §4.4: "four documented events can identify three parameters" | asserted | pending evidence, Stage 12 (identification check) | Hold until Stage 12 reports the identification and prior-sensitivity results. |
| 18 | §4.4, §3 Eq. leontief: Leontief inverse applied to direct losses | economic interpretation unstated; table type (Use vs requirements) unresolved | open, Stage 11 (SPEC §6) | After Stage 11: state which requirements table is used, the interpretation of $\bm{\Lambda}_L\bm{d}_t$, and the no-amplification comparison. |
| 19 | §4.5 Inference: joint posterior includes $z$ | joint target is valid to describe, but the marginal sampling target and recursion are unstated | continuous marginal target; emission recursion; regime paths recovered afterward (SPEC §7) | Distinguish the desired joint posterior from the marginal sampled by NUTS. In the specified non-centred representation, regime-independent Gaussian priors enter once outside the observation recursion. Sum only over observed indices and include censored endpoint masses. |
| 20 | §4.6 and Figure 1 regime alerts | historical smoothing uses later information | terminal filtered probabilities assembled across origins (SPEC §7.3) | Reserve real-time claims for verified point-in-time pipelines. Fits using later-retrieved severity or metadata remain revised-data diagnostics even at their terminal dates. Label within-fit historical curves retrospective. |
| 21 | §4.6: "Effort is carried into every forecast at its last observed value" | one projection rule | hold-last is default; trend forecast is the compared alternative (SPEC §2) | Keep; add the alternative and the Stage 10 comparison once available. |
| 22 | §4.6 trajectory initialisation, covariates, and observable forecasts | initial state and mark-existence handling unstated | initialise once, advance sequentially, preserve announced covariates, and distinguish scenarios from conditional mark scoring (SPEC §8) | Describe the terminal regime draw and conditional mark distribution. Score every eligible observed mark; do not remove cells because a finite simulation produced no positive cohorts. Score counts on all audited cells. |
| 22a | §4.4 economic calibration and uncertainty | conditional calibration distribution unspecified | cut joint with separately normalised conditionals, approximated by a finite mixture of paired draws (SPEC §6) | Give the exact integral and identify the finite mixture as a Monte Carlo approximation. Preserve regime-recovered technical/economic pairing and assess both technical and conditional simulation error. |
| 22b | §4.2: $\bm{\Phi}$ diagonal, leading block triangular (edits 5, 6) | described only as identification | a substantive restriction: no cross-factor dynamics; factor $j$ is anchored to the $j$-th densest category (SPEC §3.3) | Say so where the constraints are introduced. |
| 23 | §4.6: "Those draws are pushed through ... VaR and ES statements inherit [uncertainty from every layer]" | implies calibration and exposure uncertainty propagate; §6 of SPEC treats them conditionally | pending Stage 12 (propagation of calibration and exposure uncertainty) | Hold until Stage 12 states what is propagated. |
| 24 | §1 Introduction: forecasting scores, interval coverage, and event-range overlap | unsupported numerical results in the available workspace | pending evidence, Stages 12–13 | Replace with measured results or remove. Describe 97% versus nominal 90% as empirical overcoverage; assess sample uncertainty, dependence, interval discreteness, width, and proper scores before claiming systematic miscalibration. |
| 25 | §1 contributions: "a learned kernel that captures correlations and cross-excitation across threat topics" | the kernel is a common temporal component; cross-excitation (one category raising another later) is not in the specification | SPEC §3.5 | Replace "cross-excitation" by "shared local temporal co-movement", or add a cross-excitation mechanism. |
| 26 | §1 contributions and §4: "an open-source implementation ... that reproduces every result reported here" | no implementation exists | Stage 3 onward | Hold until true. |
| 27 | Figure 1 caption and nodes: $\bm{Q}$, $\sigma_g$, $w_d(h)$, $\Pr(z_t = r \mid \mathcal{F}_T)$, "$B \sim$ LogNormal", "$E$ exploitation signal" single channel | | as above | Propagate edits 2, 5, 9, 12, 20 into the figure. |
| 28 | §3 Problem Formulation: exposure from NVD CPE fields | no coverage or historical-version audit | audit required (SPEC §1; Stage 11) | Report measured CPE coverage and handling of uncovered CVEs. Distinguish NIST's enrichment prioritisation from publication completeness; do not infer a historical coverage percentage from policy announcements. |
| 29 | §4.2 prior description | priors claimed to encode no beliefs; noise floors conflated with stochastic transitions | explicit proposed scales and prior-predictive checks | A Half-Normal noise floor affects residual variation, not the Markov transition matrix. Positive Dirichlet row parameters already imply strictly positive transitions almost surely. All priors can influence inference. |
| 30 | Economic outputs | loss units, operating-point uncertainty, and feasibility limits unspecified | conditional gross-output accounting with an explicitly fixed operating point (SPEC §6) | Distinguish gross-output loss from GDP/value added, disclose monthly-output approximation and plug-in normalisation, and report sectors lacking a calibrated positive reference. |
| 31 | Future experimental sections | no implemented benchmark | Stage 2 and SPEC §9 specify eligibility and scoring | Freeze target-month eligibility, conditional mark targets, revised-data versus as-of experiments, and failure policies. Do not claim independent economic validation using calibration events. |

## Verified source facts used above

| Fact | Source |
|---|---|
| EPSS archive and model-release dates in SPEC §1; historical score distributions change across versions | [FIRST data page](https://www.first.org/epss/data.html); [maintainer archive](https://github.com/empiricalsec/epss_scores). Individual gaps remain an ingestion-audit task. |
| KEV launched on 3 November 2021 with a backlog; official fact sheet gives approximate components, not the asserted exact 306 | [CISA launch fact sheet](https://www.cisa.gov/sites/default/files/publications/Reducing_the_Significant_Risk_of_Known_Exploited_Vulnerabilities_20211103.pdf) |
| SEC effective date 5 September 2023; reporting compliance dates 18 December 2023 and 15 June 2024 | [SEC release](https://www.sec.gov/rule-release/33-11216); [SEC small-business compliance guide](https://www.sec.gov/resources-small-businesses/small-business-compliance-guides/cybersecurity-risk-management-strategy-governance-incident-disclosure) |
| NVD enrichment policy does not imply all unprioritised records lack metadata or that publication counts are complete | [NIST announcements](https://www.nist.gov/itl/nvd); empirical coverage remains to be measured |
| NVD exposes a CVE Change History API; reconstruction completeness is not established by its existence | [NVD team's API announcement](https://groups.google.com/a/list.nist.gov/g/nvd-news/c/8FGMfqB2PQ8) |
| Censored endpoints require probability masses, not continuous density values | [PyMC censored distribution](https://www.pymc.io/projects/docs/en/v5.16.0/api/distributions/censored.html) |
| Kernel lag-12 error is below $10^{-4}$ at the four checked widths; lag-34 error at $b=18$ is approximately $2.3843\times10^{-4}$ | [kernel_check2.py](checks/kernel_check2.py) and [recorded output](checks/kernel_check2.out.txt); finite checks are not a uniform bound |

## Items the specification leaves as in the manuscript

- Markov-switching mean (levels only), not switching VAR (§4.2).
- Triangular loading constraint with positive diagonal (§4.2), supplemented by edit 5.
- A shifted Half-Normal floor for idiosyncratic intensity noise; other scale priors are not universally floored.
- $\pi_s$ constant within the incident-rate expression; the reporting-population multiplier changes its contribution to the total rate.
- Masking rather than imputation for empty category-months (§4.3).
- NUTS on the continuous posterior; FFBS regime recovery after sampling (§4.5).
- Hold-last effort projection as the default forecast rule (§4.6).
- Twelve-month horizon, four outputs (§4.1, §4.6).

## Stage 5 review carried into implementation (2026-10-20)

- Describe the learned categories as vulnerability-description topics. Three of
  the nine known topics prominently reflect Oracle/Adobe/template wording;
  they are not established independent mechanisms of malicious activity.
- Distinguish stable panel IDs from loading-matrix row order. The initial-count
  order is 8, 2, 6, 1, 4, 9, 3, 5, 7, then unknown (10).
- Keep unknown in the primary count score, and report its contribution alongside
  the active stratum. Its observed zero counts inform the count likelihood; a
  near-zero error on this fallback series is not substantive forecasting evidence.
- Preserve the 21 unmatched KEV additions. Version G must explicitly choose
  masking unknown in its KEV likelihood or a separate intercept-only rate that
  does not depend on the count-linked intensity. The choice remains open.

These requirements are also recorded in [the carry-forward note](STAGE6_8_CARRY_FORWARD.md).

## Stage 6 control implementation (2026-10-21)

- Distinguish the five initial controls from the eight named comparison systems.
  The active independent control is a conditional Bayesian log-linear NB
  autoregression (`independent_nb_glar`), not the unsuccessful latent AR(1) model.
  Cite its [dated amendment and sources](amendments/OBSERVATION_CONTROL_20261021.md).
  State the conditional likelihood, first-count initialization, empirical
  initial-window prior anchor and recursive predictive law. Its numerical priors
  are project choices. The failed latent fits remain computational evidence.
- Describe the approved all-zero ARIMA/ETS point-mass convention explicitly.
  Keep all categories and disclose the fallback category's effect on averages.
  Do not imply that ETS has been implemented merely because its boundary rule exists.
- Report Stage 6 scores as development results. ARIMA is selected on development
  CRPS; this does not establish held-out superiority. MAE uses predictive medians
  and RMSE uses predictive means. Count log scores evaluate issued PMFs; no
  winsorization or moment-matched density is substituted for a forecast law.
- Keep control-posterior validation distinct from proposed-model recovery,
  matched factor/regime ablations, held-out calibration and economic validation.
  The [Stage 6 record](STAGE6_COMPLETION.md) identifies completed evidence and
  remaining stages; all old failed-evaluation numerical claims still require replacement.

## Stage 7 prior and coordinate changes (2026-10-21)

These changes are active for synthetic validation under
[the activation record](amendments/STATE_SPACE_PRIORS_ACTIVATION_20261021.md).
They do not establish successful recovery or authorize held-out fitting.

- Supersede edit 5's identity innovation covariance with
  `Q = diag(1 - phi_j**2)` and stationary initial factors `N(0,I)`.
  Each factor has unit stationary variance; loadings measure stationary
  amplitude. Apply the same convention to severity factors and forecasts.
  This is a change of prior law relative to Stage 2, not merely coordinates.
- At `main.tex:552`, the figure node near line 357, and the figure caption
  at line 518, replace the old scale/loading descriptions with the actual
  normalized truncated laws in the amendment. Free loading entries have
  scale `1/sqrt(r)` and bounds `+/-2/sqrt(r)`; leading diagonal entries are
  positive. Optional iid count SD has a HalfNormal(1) prior truncated to
  `(0,1)`, with no artificial 0.01 floor. Supersede the retained-floor bullet
  in the historical list above. Do not claim these restrictions encode no
  beliefs, or that a noise floor makes Markov transitions stochastic.
- Distinguish these substantive restrictions from the equivalent sampling
  coordinates in [the geometry decision](decisions/STAGE7_AMPLITUDE_COORDINATES_20261021.md).
  Scaling factor states by their anchor loading, with the full transition
  normalizers/Jacobian, preserves the amended probability law.
- Finite prior moments do not establish plausible predictive spread,
  posterior convergence, recovery, or calibrated forecasts. Report those
  checks separately, including failures and sensitivity results.
- The proposed training-only level anchor, narrower level hierarchy, and
  independent fallback-category level are under review. They are not the
  laws of the already running `stationary_bounded_v1` fits. A subsequent
  amendment must specify the exact anchor, eligible categories, reference
  effort convention, and treatment of synthetic fixtures before new fits.
- Version G's bounded KEV elasticity is a separate substantive restriction.
  Count-only validation does not validate that slope law or resolve the
  unmatched-KEV cohort issue.

### Second Stage 7 amendment: levels and fallback

The [anchored-level amendment](amendments/STATE_SPACE_LEVELS_20261021.md)
supersedes the pending-review status above for synthetic validation. State
`m_bar ~ N(c,1)`, `s_m ~ HalfNormal(1)` truncated to `(0,2)`, and the independent
fallback level `N(log(mean_initial_fallback+0.5),1)`. For the real-panel design,
the known-only initial-window anchor is 4.301907835285142. This is data-informed
regularization; it is not an externally elicited prior or an exact conversion
from an observed mean to a reference-effort latent median. Synthetic experiments
compute their own anchors without real counts or future observations.

Report reference-effort and longitudinal prior checks separately from stress
envelopes, posterior convergence and recovery. Include both tightened and
widened sensitivity laws, and disclose the fallback's remaining dependence
through shared dynamics. Keep all count categories in primary scoring.

### Severity sampling coordinates

The [conditional severity coordinates](decisions/STAGE7_SEVERITY_COORDINATES_20261021.md)
use a Gaussian working conditional to transform the latent states. The model
still evaluates the censored mark likelihood and the normalized stationary
state prior, including the parameter-dependent transformation Jacobian.
This changes the sampling coordinates, not the probability law. Independent
precision, complete density-ratio and gradient checks passed, including missing
marks and censored endpoints. Report posterior diagnostic outcomes separately;
passing these identities alone does not establish successful inference.

### Count coordinates and numerical implementation (2026-10-22)

The [conditional count coordinates](decisions/STAGE7_COUNT_COORDINATES_20261021.md)
likewise use a working Gaussian approximation solely to choose an invertible
transformation. Counts retain their exact negative-binomial likelihood,
including observed zeros; missing cells supply no proxy information. The
normalized state density and parameter-dependent Jacobian remain in the target.
For the validated one-regime submodel without iid noise or kernel, block
Cholesky and triangular solves evaluate the same transformation as the dense
reference. The cached implementation supplies exact first derivatives. It is
not a Gaussian replacement model or an approximate posterior method.

Document the independent dense-algebra, finite-difference and fused sampler
log-density/gradient tests. The fused interface tests caught a scalar-versus-
Tensor return error missed by separate numerical tests; the failed launches
are preserved as execution errors. Keep these distinct from diagnostic failures.
The [metric comparison](decisions/STAGE7_WHITENED_METRIC_20261022.md) changes only
mass adaptation after whitening and must be reported with its actual outcomes.
Timing evidence separates compilation, warmup and retained sampling, records
compiler optimization and concurrency, and is not a real-data rolling-fit budget.

### Simulation denominators and interpretation (2026-10-22)

Describe the current design as 43 primary scenario/seed cases and 12 sensitivity
cases, not 43 independent datasets. The effort variants deliberately share
observed counts, and coordinate retries reuse both observations and generating
truth. The final audit fingerprints observed panels and distinguishes canonical
fits from preserved earlier attempts. Do not add duplicate verification of the
same severity files to the number of fitted cases.

Separate numerical correctness, convergence, fixed-truth recovery and predictive
calibration. The [fresh full suite](checks/stage7-final-tests-20261022/audit.json)
passes 222 tests across 29 files; it establishes the first category only.
Only complete posteriors passing all prescribed gates with no unresolved
warnings contribute to recovery summaries. Recorded, recognized performance
warnings are distinct from unresolved warnings. Failed or interrupted chains cannot substantiate a
component's benefit, and three fixed-truth seeds are not SBC. Retain failure
denominators in tables rather than reporting only admitted fits.

The real-data anchor uses the known categories in 2010--2018; it is empirical-
Bayes regularization. The wide-level prior fails its plausibility gate and the
wide-dynamics prior remains unresolved after its declared repeat. Label their
posterior comparisons as sensitivity stress cases, not additional admitted
priors. Report all sensitivity outcomes without choosing the best recovery seed
or candidate. Final posterior findings belong in the completed Stage 7 report.

Counts alone identify the sum of log intensity and reporting effort, not their
physical decomposition. A count-derived effort estimate supplies a reference
scale; convergence and count accuracy do not prove that it separates actual
malicious activity from disclosure throughput. Similarly, fitted regime
probabilities using the complete synthetic history are retrospective and do not
validate an early-warning claim. Preserve these distinctions in the abstract,
methodology and eventual evaluation discussion.

### Noise separation and posterior exploration (2026-10-22)

The [iid conditional-coordinate decision](decisions/STAGE7_IID_COORDINATES_20261022.md)
is a computational change of variables, not a new prior or likelihood. The
six conditional noise-separation fits pass numerical diagnostics, but sparse
observations leave log dispersion close to its prior. Report the positive
tau/log(psi) tradeoff and the favorable fixed-level conditioning explicitly;
see the [noise analysis](simulations/stage7-noise-20261022/README.md). This supports
keeping iid noise optional and the initial pilot parsimonious, not a claim
that its two noise components are generally identified.

Numerical admission alone is insufficient when an earlier retained attempt
reveals another orientation that an admitted retry does not explore. The
[recovery-eligibility decision](decisions/STAGE7_RECOVERY_ELIGIBILITY_20261022.md)
requires separate numerical and recovery counts, with evidence-bound exclusions.
Do not call the existing proxy or kernel mode-conditional recovery a verified
marginal posterior result. The [working sign chart](decisions/STAGE7_SIGN_GAUGE_20261022.md)
preserves the rank-one target and canonical positive-first-loading convention;
it changes the smooth sampling coordinates, not the reported model or category
order. Its actual exploration results must be reported after the verification
and refits, without inferring mode probabilities from the old chain split.

### Joint coordinates, grid completion and compiler disclosure (2026-10-22)

The exact joint level/factor transformation uses a Schur complement and retains
the original normalized priors and Jacobian. It changes coordinates, not the
NB likelihood or prior. Record the [verification and scheduling decision](decisions/STAGE7_SYNTHETIC_GRID_COMPLETION_20261022.md),
including superseded interrupted trials and failure denominators. Removing a
synthetic scheduling dependency does not admit any posterior or real-data fit.
The representative fixture matches panel dimensions but contains a deliberately
pooled sparse category; the fallback fixture tests the separately declared
independent fallback level. They must not be described as actual panel fits.

The generated PyTensor functions retain their default selected fast-math flags;
the custom block kernels use fastmath=False. State this explicitly instead of
claiming fast-math is globally disabled. The compiler profile check compares
complete targets/gradients with the native calculation and discloses first-call
and compile overhead. The narrowly recognized dynamic-global cache warning
means that a compiled function cannot be written to disk; it is preserved as a
performance warning, not silently treated as a numerical error or discarded.

The [reviewed effort experiment](simulations/stage7-effort-reviewed-20261022/README.md)
supersedes the earlier mode-conditional summary. Eight of nine fits pass the
numerical gates, but only six are recovery eligible: proxy seeds 7101 and 7103
do not explore documented alternative canonical regions. Only proxy seed 7102
supports the paired three-configuration recovery comparison. Do not omit these
exclusions or convert its count accuracy into a physical reporting-effort claim.
