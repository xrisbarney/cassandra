# Model corrections vs. the paper (read before finalizing the manuscript)

While making the model actually sample, three issues in the paper's stated
specification (Section 3, Eqs. 3–8) had to be corrected. The first two are
**genuine specification bugs** — the model as written is not identified and will
not sample — so the manuscript's equations should be updated to match. The third
is an over-parameterization that wrecks the sampling geometry. Everything else is
an implementation detail with no effect on the stated model.

---

## 1. Factor loadings Γ are not identified — **update Eq. (3)**

**Paper:** `η_t = μ_{z_t} + Γ f_t + u_t`, with a free `K×r` loading matrix `Γ` and
free factor innovation covariance `Q_{z_t}`.

**Problem:** this is the classic dynamic-factor-model non-identification. For any
invertible `r×r` matrix `R`, the triples `(Γ, f_t, Q)` and
`(ΓR, R⁻¹f_t, R⁻¹QR⁻ᵀ)` give an identical likelihood — the loadings and factors
are jointly determined only up to rotation, scale, and sign. The posterior has
flat ridges; NUTS responds by collapsing its step size to ~1e-10 and produces
**no usable samples** (the chain does not move).

**Fix (in code):** constrain the top `r×r` block of `Γ` to be lower-triangular
with a positive diagonal (rows below stay free). This is the standard PLT
identification restriction and removes the rotation/scale/sign freedom.

**Manuscript change:** state this constraint where `Γ` is introduced, e.g.
"For identification, `Γ` is constrained so that its leading `r×r` block is
lower-triangular with strictly positive diagonal entries."

## 2. Severity loadings Ψ — same fix

The latent severity process (`log σ_{k,t} = ν_{z_t} + Ψ h_t + v_t`) has the same
free-loading non-identification. The code applies the same PLT constraint to `Ψ`;
the manuscript should state it if the severity factor process is described.

## 3. Reporting propensity π should not be free per sector-month — **reconsider Eq. (8)**

**Paper:** `π_{s,t}` (a per-sector, per-time reporting propensity) enters the
incident channel `D_{s,t} ~ Poisson(π_{s,t} · ρ_s · Σ_k M_{s,k,t} λ_{k,t})`.

**Problem:** treating `π_{s,t}` as free for every sector *and* every month is
`S×T ≈ 2000` nuisance parameters on a channel that is ~99% zeros (Item 1.05
disclosures barely exist before 2024). The data pin almost all of them against
zero, producing a boundary funnel that — on its own — collapses the sampler.

**Fix (in code):** one baseline per sector, constant over time (`π_s`, length S),
broadcast over months.

**Manuscript change:** define `π` as sector-specific and time-invariant (or, if
time variation is wanted, as a smooth latent process — e.g. a random walk in
`logit π_{s,t}` — rather than free per-month parameters). The current wording
implies free per-month values, which is not estimable from the data.

---

## Implementation details (no manuscript change needed)

These make the *stated* model sample well; they don't change its meaning:

- **Non-centered idiosyncratic noise.** `u_t` and `v_t` (and the two severity
  innovations) are sampled as `standardized ε ~ N(0,1)` then scaled
  (`u = τ·ε`), instead of directly from `N(0, τ²)`. Removes Neal's-funnel
  geometry. Mathematically identical model.
- **Stationary AR.** The factor/severity AR matrices are passed through `tanh`
  before the lower-triangular mask, bounding eigenvalues to (−1, 1) so factors
  cannot explode over the ~180-month series.
- **Smooth overflow saturation.** Log-intensities are passed through
  `30·tanh(x/30)` (not a hard `clip`) before `exp`, so the Negative-Binomial
  mean stays finite while the gradient never vanishes — a hard clip has a
  zero-gradient plateau beyond its bounds, which is a known source of leapfrog
  instability if a trajectory's momentum ever carries it past the bound.
- **Scale floors.** `Q_r`, `tau_k`, `Q_h`, `omega_k`, `kappa_k`, and
  `varsigma_k` are `0.02 + HalfNormal(...)` rather than plain `HalfNormal`, so
  a state transition can never become exactly deterministic (infinite
  curvature) under the prior.
- **Data-informed initialization.** `mu_r` (baseline log-intensity per
  regime/topic) starts near the empirical per-topic log-count level instead of
  the prior median (0). Real topics span multiple orders of magnitude (e.g.
  mean count 25/month vs. 450/month); starting every topic at 0 means the
  first few leapfrog steps face a very badly scaled gradient. `mu_r`'s prior
  was also widened (`sd`: 2 → 5) to accommodate this range.
- **NaN masking.** Topic-months with no CVEs carry NaN in the EPSS and
  severity channels; those are masked out of their likelihoods.

## Result

With items 1–3 fixed, the adapted NUTS step size goes from ~`3.6e-10` (no
sampling at all) to ~`1e-3`–`1e-2` (healthy) on the real 180-month panel, and
out-of-sample calibration is good (empirical coverage ≈ nominal coverage;
PIT not significantly non-uniform) once a run adapts successfully.

**However, this specific model + real-data combination — NUTS wrapped in
`DiscreteHMCGibbs` for the discrete regime path, on the actual 180-month,
K=8/S=11/R=3 panel — did not adapt successfully for every random seed.**
`numpyro`'s own documentation flags `DiscreteHMCGibbs` as an
`[EXPERIMENTAL INTERFACE]`, and empirically a large fraction of seeds (12/13
in one session's testing) caused the warmup step-size adaptation to collapse
to numerical underflow (~1e-38) at some point during warmup — seemingly a
property of that seed's specific HMC/Gibbs trajectory rather than a further
fixable structural defect, since it happened unpredictably with every one of
the mitigations above individually applied. A retry-with-new-seed safety net
in `run_nuts` did not reliably rescue this (a meaningful fraction of seeds
failed even across 6 consecutive retries).

## 4. Discrete regime path is marginalized analytically, not Gibbs-sampled — **the actual fix**

`DiscreteHMCGibbs` was replaced entirely. `full_model` (in
`src/cassandra_threatcast/model/full.py`) now marginalizes the discrete
regime `z_t` out of the likelihood analytically, via the standard HMM forward
algorithm: at each scan step, the four channels' log-likelihoods are computed
under each of the `R` regime hypotheses (via `jax.vmap`, using plain
`.log_prob()` calls rather than `numpyro.sample` statements), and combined
into a running forward-filter vector `log_alpha_t[r] = logsumexp_{r'}
(log_alpha_{t-1}[r'] + log Pi[r',r]) + loglik_r[r]`. The total marginal
log-likelihood `logsumexp_r(log_alpha_T[r])` is injected once via
`numpyro.factor(...)`. `z_t` is never sampled — there is nothing for a Gibbs
kernel to update — so plain `NUTS` (in `src/cassandra_threatcast/inference/
nuts.py`) now samples the entire continuous parameter space directly.

This was prototyped first with `numpyro.contrib.funsor`/`@config_enumerate`
(automatic enumeration), which numpyro's docs recommend as the standard
alternative to `DiscreteHMCGibbs`. That path got as far as tracing cleanly
(after wrapping ~15 multi-dimensional prior sites in explicit
`numpyro.plate`s and switching regime-indexed lookups to
`numpyro.ops.indexing.Vindex`), but broke down structurally once the discrete
regime affected the continuous factor's *transition* matrix
(`Phi_{z_t}`/`Q_{z_t}`): the scan's carry becomes enumeration-dependent and
its shape becomes inconsistent across iterations (`dot_general` shape errors,
and "joint log density expected scalar, got (R,R)" when `z_t` is reused for
more than one downstream emission in the same step). This is a genuine
structural limit of automatic enumeration over a **switching linear dynamical
system** (a discrete state that drives the transition dynamics, not just the
emission), not a bug to work around — so it was abandoned in favor of the
hand-written forward algorithm above, which has no such restriction because
there is no enumeration machinery involved at all.

**Consequence — a further, transparent model simplification (update Eq. (3)
if reported in the manuscript):** to make marginalization tractable, the
factor and severity AR transition dynamics are now **regime-constant**
(`Phi`, `Q_f`, `A_h`, `Q_h` — no `R` axis); only the emission *means*
(`mu_r[z_t]`, `nu_r[z_t]`) still switch by regime. This is a "Markov-switching
mean" model rather than the paper's full "Markov-switching VAR"/switching
linear dynamical system. `A_h`/`Q_h` were already regime-constant in the
original code, so in practice this only changes `Phi_r → Phi` and
`Q_r → Q_f`. If the manuscript states regime-dependent factor transition
dynamics, it should be revised to state that only the level switches by
regime, or this should be flagged as a documented deviation between the
paper and the implementation.

**Validation:** at the real data's scale (K=8, S=11, T=180, r=3, r_sig=2,
R=3), plain NUTS on the new model sustained healthy step sizes (~2e-3 to
5e-2) across 30+ warmup/sampling iterations with no collapse and
acceptance probabilities in the 0.81–0.92 range, on realistically-scaled
synthetic data across multiple seeds — matching or exceeding the best cases
previously seen with `DiscreteHMCGibbs`, without that approach's frequent
failures. The `run_nuts` retry-on-collapse loop is kept as a cheap safety
net but is not expected to be needed in normal operation.

**Downstream consequence for forecasting:** `predict()` no longer reads a
literal sampled `z_t`; it recovers the *exact* filtered posterior over the
terminal regime via the Hamilton forward filter
(`src/cassandra_threatcast/inference/ffbs.py::forward_filter`, applied to the
per-draw `loglik_regime_t` site the model now exposes) and samples
`z_T ~ Categorical(filtered_probs_T)` to seed the forward simulation. `f_t`/
`h_t` are now simple, non-branching sequences (no per-regime dispatch), so
their terminal values are directly usable without any special recovery.

## 5. Out-of-sample backtest against real 2025 data — severe under-prediction, poor calibration

The model was trained on 2010-01 through 2024-12 and forecast 12 months
ahead (2025-01 through 2025-12). 2025 has since happened, so
`scripts/backtest.py` fetches the real, now-known 2025 CVE data (classified
into topics via the *frozen*, not refit, topic mapper, so topic indices stay
aligned with the trained model's per-topic parameters) and scores the
original forecast against it. This is the same 45-day-cadence
rolling-origin methodology as `evaluate.py`, but against genuinely
out-of-sample future data rather than a held-out slice of the training
window.

**Result: the forecast significantly under-predicted actual 2025 CVE
volume, and its stated uncertainty intervals were badly overconfident.**

| Threat category (AI-assigned name) | Predicted (sum, 2025) | Actual (sum, 2025) | Actual / predicted |
|---|---:|---:|---:|
| Cross-Site Request Forgery | 617 | 9,348 | 15.2x |
| Miscellaneous Vulnerabilities | 2,047 | 10,565 | 5.2x |
| Denial of Service | 1,880 | 8,744 | 4.7x |
| Information Disclosure | 518 | 2,043 | 3.9x |
| Cross-Site Scripting | 559 | 2,159 | 3.9x |
| SQL Injection | 1,214 | 4,166 | 3.4x |
| Remote Code Execution | 6,096 | 10,368 | 1.7x |
| **Total** | **12,931** | **47,393** | **3.7x** |

Empirical 90% predictive-interval coverage across all topic-months was
**4.2%** (target: ~90%) — the actual value fell inside the model's stated
90% range in only 2 of 96 topic-months. This is not a borderline
miscalibration; the model's uncertainty quantification cannot currently be
trusted for genuine future forecasting, even though in-sample calibration
(the "Result" section above, and `evaluate.py`'s rolling-origin folds) looks
reasonable.

**This was chased down and confirmed to be a real forecasting-accuracy
finding, not a units/scale bug in the backtest** (verified by comparing
against the training panel directly): the training data's *own* final year
already shows the acceleration the forecast failed to extrapolate. The
15-year (2010-2024) monthly mean is ~1,226 CVEs/month; the last 12 months of
training data (2024) alone average ~3,310/month — already a ~2.7x jump
within the training window itself — and the real 2025 rate (~3,950/month)
simply continues that trajectory. The forecast, however, reverted toward
something closer to the 15-year historical average (~1,078/month predicted)
rather than continuing the acceleration already visible in the most recent
training data.

**Mechanism — diagnosed, and mostly a concrete implementation defect rather
than a deep modelling limitation.** Two effects compound; the first is by
far the larger.

*Primary cause (≈ the whole gap): the forecast silently drops the
reporting-effort offset `e_t`.* In training, the CVE-count channel's mean is
`μ = exp(soft_clip(η_t + e_t))` (`full.py:260`) — `e_t` is a fixed additive
log-offset with coefficient 1, the HP-filter trend of log-total-CVE-counts,
standardized to mean 0 / std 1 over the panel. Because counts trended up,
`e_t` runs from ≈ −1.2 (2010) to ≈ +1.85 (2024-12). But `predict()`
(`full.py:489`) computes `μ = exp(η)` with **no `e_t` term at all** — i.e.
it implicitly assumes `e_t = 0` (the 15-year *average* effort) for every
forecast month, which asserts that reporting/discovery effort instantly
snaps from its 2024 peak back to the 2010-2024 mean in January 2025. Since
`η` is fit to explain `log N − e_t`, exponentiating `η` alone under-counts
by ≈ `exp(e_t_end)`. Quantitatively (offline sensitivity check against the
2025 actuals, first-order in log-space):

| Assumed forecast `e_t` | Forecast total | actual / forecast |
|---|---:|---:|
| 0 — current behaviour | 12,931 | 3.66× (under) |
| +1.30 — implied best-fit | ≈ 47,400 | ≈ 1.0× |
| +1.70 — hold 2024 12-mo mean | 70,718 | 0.67× (over) |
| +1.85 — hold last value | 82,443 | 0.57× (over) |

So dropping `e_t` explains essentially the entire 3.66× under-prediction:
the actuals correspond to an *effective* offset of +1.30 — elevated well
above the 15-year mean (the model used 0) but a little below a naive hold of
the 2024 smoothed-trend peak (+1.70). That the best-fit value sits below the
hold-last value is consistent with (i) 2025 discovery growth decelerating
slightly vs. 2024, and (ii) **HP-filter endpoint bias** — a two-sided
smoother systematically over-/under-shoots at the sample boundary, exactly
where a forecast originates.

Note that `predict()` *does* hold the exposure map `M_future` constant at its
last observed value; dropping `e_t` while holding `M` constant is almost
certainly an oversight rather than a deliberate modelling choice. **Minimal
correct fix:** project `e_t` forward into `predict()` (at least held
constant, better a damped/one-sided forecast — see §6) instead of implicitly
zeroing it.

*Why in-sample evaluation (`evaluate.py`) did not catch this:* the same
`e_t` drop affects the rolling-origin folds, but those folds' forecast
windows sit where `e_t ≈ 0` (fold `train_T=72`→ mean `e_t` −0.54, rising to
+1.52 by `train_T=162`). Dropping a ≈0 offset barely bites, so the defect is
nearly invisible for early/mid-sample folds and only fully bites the genuine
end-of-series 2025 forecast where `e_t` peaks. This is a cautionary example
of an out-of-sample failure that in-sample cross-validation structurally
cannot surface, because the very quantity that breaks (a large terminal
`e_t`) does not occur inside the cross-validation windows.

*Secondary cause (compounding, but small next to the above): AR mean-
reversion.* The factor/severity AR dynamics are constrained stationary
(`tanh`-bounded eigenvalues, see "Implementation details") so the sampler
doesn't diverge over 180 months. A stationary AR is mean-reverting, so over
a 12-month horizon `η` itself drifts toward its long-run level rather than
extrapolating a trend. With `e_t` correctly restored this residual effect is
modest, but it is the reason even the `e_t`-corrected forecast would not
track a *continuing acceleration* — only a sustained *level*.

*Separate, still-open problem — overconfidence (the 4.2% coverage).* Fixing
the level bias (`e_t`) shifts the predictive *mean* up toward the actuals but
does not by itself widen the predictive *intervals*; 4.2% coverage says the
intervals are far too narrow regardless of where they are centred. This is a
distinct dispersion/calibration issue (see §6) and must be addressed
separately from the level bias.

**Recommendation for the manuscript:** report this backtest alongside the
in-sample calibration numbers, and be explicit that in-sample calibration
looking reasonable does *not* imply out-of-sample reliability here — the
dominant failure (the `e_t` drop) is invisible in-sample by construction.
Concrete remedies are catalogued in §6; the single highest-leverage one is
simply carrying `e_t` into the forecast.

**Caveat on scope:** the dollar-loss forecast could not be backtested the
same way — there is no routinely-collected "actual economy-wide cyber loss"
series to compare against (the damage-function calibration in `economic.py`
uses a handful of named historical mega-breaches, not an ongoing series).
The backtest instead validates the loss model's real drivers (CVE counts
above, and disclosed-incident counts via SEC 8-K Item 1.05 filings, when
that data source is available — it was down at the time of this backtest).
Given the CVE-count under-prediction above, the loss forecast is very
likely under-predicting actual 2025 losses by a similar multiple, though
this can't be confirmed directly.

## 6. The CVE surge, the "AI-accelerated discovery" hypothesis, and how to tune the model for it

The §5 backtest raises a substantive question: *why* did CVE volume roughly
triple over the training window (≈1,226/month over 2010-2024 vs. ≈3,310/month
in 2024 and ≈3,950/month in 2025), and does the model's treatment of that
growth make sense? A proposed explanation is that the maturation of agentic
AI, AI-assisted red-team tooling, and rising human security-research capacity
has accelerated the *discovery* of vulnerabilities. This section assesses
that hypothesis honestly and lists concrete, architecture-compatible tuning
options. **These are analysis and design suggestions for the manuscript, not
changes that have been made to the code.**

### 6.1 Is the AI-accelerated-discovery hypothesis plausible?

Plausible as a *contributing* driver — but it should be stated as one
hypothesis among several, not asserted as the cause, because our data cannot
causally isolate it and several better-documented drivers overlap the same
period. Candidate drivers of the 2023-2025 surge, roughly from best- to
least-established for that window:

1. **CNA (CVE Numbering Authority) expansion.** MITRE sharply grew the number
   of organizations authorized to assign CVE IDs. This mechanically raises
   recorded counts independent of any change in the underlying vulnerability
   rate — arguably the single largest, most certain contributor to the raw
   count growth.
2. **NVD process turbulence (2024).** NIST's analysis/enrichment backlog in
   2024 disrupted the cadence and completeness of records; this affects the
   enriched channels (CVSS `B`, EPSS `E`) more than raw counts, but signals
   that the count series is partly a process artifact.
3. **Secular growth in security research** — more bug-bounty programs, more
   funded researchers, wider disclosure norms, and larger attack surface
   (more software, dependencies, IoT).
4. **Automated discovery tooling** — large-scale fuzzing (e.g. OSS-Fuzz),
   static analysis, and increasingly **AI/LLM-assisted and agentic
   discovery** (LLM-guided fuzzing, autonomous vulnerability-finding agents).
   This is the newest driver and the one the hypothesis names; it is
   plausibly real and *growing* into 2025, but it is also the hardest to
   isolate and, for the specific 2024 jump, likely smaller than (1).

Two honest caveats the manuscript should carry:

- **Attribution is not identified.** Nothing in the data lets us apportion
  the trend among (1)-(4). Any AI-specific claim is therefore
  hypothesis-generating, not causal. The defensible phrasing is: "the
  discovery/reporting-effort trend accelerated markedly in 2023-2025,
  consistent with — among other drivers — the maturation of automated and
  AI-assisted vulnerability discovery."
- **Discovery rate ≠ realized risk.** The model forecasts *recorded CVE
  counts*, which conflate the true vulnerability-generation rate with the
  rate at which vulnerabilities are *found and catalogued*. If AI mainly
  accelerates discovery of pre-existing flaws, a count surge partly reflects
  latent risk being *surfaced* (and often patched) faster, which does not map
  one-to-one onto increased *realized* economic loss. This matters for the
  Leontief loss layer: scaling losses off a discovery-inflated count risks
  overstating systemic loss. The channels closest to realized risk are the
  incident channel `D` (SEC 8-K Item 1.05) and exploitation signals
  (EPSS `E`, CISA KEV) — the manuscript should lean on those, not raw CVE
  counts, when making loss claims, and should explicitly separate "we
  forecast more *disclosed vulnerabilities*" from "we forecast more *harm*."

### 6.2 How the model already encodes this — and where it goes wrong

The reduced-form home for all of drivers (1)-(4) is already in the model:
the **reporting-effort covariate `e_t`** (`features/effort.py`), the
HP-filter trend of log-total-CVE-counts entering as a fixed additive
log-offset. It is, by construction, a catch-all "detection/reporting
capacity" trend — it absorbs whatever combination of CNA growth, research
intensity, and AI tooling drove the low-frequency count level, without
disentangling them. So the model *does* have a mechanism for the surge; the
§5 failure is not that the mechanism is missing but that (a) the forecast
discards it (`predict()` implicitly sets `e_t = 0`), and (b) even used
correctly, how `e_t` is *projected* forward is an unmodelled choice. There is
also a mild circularity worth disclosing: `e_t` is estimated from `N` and
then `N` is modelled conditional on `e_t`, so the low-frequency level is
effectively supplied to the model as a known offset and the forecast's
accuracy hinges almost entirely on the `e_t` projection.

### 6.3 Tuning options (architecture-compatible), grouped by the problem they fix

**A. Level bias — fix the `e_t` projection (highest leverage; see §5).**
- *Minimal:* carry `e_t` into `predict()` and hold it at its last observed
  value over the horizon, exactly as `M_future` holds exposure constant.
  Per §5 this over-shoots slightly (endpoint bias), but converts a 3.7×
  *under* into ≈1.5× *over* — a large net improvement and an easy, defensible
  default.
- *Better:* replace the two-sided HP filter with a proper one-sided /
  state-space local-linear-trend for `e_t` (the code already exposes a
  `state_space` `UnobservedComponents(level="local linear trend")` option in
  `effort.py` — currently unused by the default `hp_filter` path). A
  state-space trend produces a principled *forecast* of `e_t` with its own
  uncertainty, avoids HP endpoint bias, and lets the forecast damp the trend
  rather than either freezing or naively extrapolating it.
- *Best (and most aligned with the AI hypothesis):* model `e_t` jointly
  *inside* the Bayesian model as a latent trend with drift, rather than as a
  pre-computed fixed offset. Then its forecast uncertainty propagates into
  the count intervals (helping problem C below), and one can attach an
  interpretation/prior to its drift.

**B. Structural trend / non-stationarity (so the model can extrapolate a
sustained rise, not just a level).**
- Add a **local-linear-trend or random-walk-with-drift** component to the
  factor level (or directly to `η`), coexisting with the stationary AR that
  handles short-run dynamics. This is the standard structural-time-series
  decomposition (trend + cycle) and directly addresses the mean-reversion
  noted in §5. It must be introduced carefully — an unbounded integrated
  component reintroduces the divergence risk the `tanh` stationarity
  constraint was added to prevent — so it should carry a tight
  half-Normal prior on the drift/innovation scale.
- **Stickier regimes.** Give the transition matrix `Pi` an asymmetric
  Dirichlet prior with a large diagonal concentration (persistence prior) so
  a newly-entered high-volume regime *stays* through the forecast rather than
  decaying toward the stationary distribution. Also worth inspecting: the
  forecast evolves the regime forward via `Pi` from the filtered terminal
  state — check (by dumping the fitted `Pi` and the 2023-2024 filtered regime
  probabilities) whether the high-volume regime was actually identified as
  dominant at the cutoff or was underweighted.

**C. Overconfidence — widen predictive dispersion (the 4.2% coverage).**
- Turn on/evaluate **enhanced mode** (Student-t latent innovations, NegBin
  incident channel) which exists precisely to fatten tails.
- Revisit the **scale floors and priors** on `Q_f`, `tau_k`, `psi_k`
  (NegBin concentration): overly-tight innovation scales or too-high
  concentration produce intervals that are too narrow. Consider letting
  forecast-horizon innovation variance grow with `h`.
- Ensure **full parameter uncertainty** propagates: `predict()` already loops
  over posterior draws (good), but confirm each trajectory also redraws the
  regime path and innovations, and — if `e_t` is moved inside the model (A,
  best) — that its forecast uncertainty flows through too.

**D. The "just use the last 1 year" question — no, but the intuition is
right.** Literally training on 12 monthly observations cannot identify this
model: with `K=8`, `S=11`, `R=3` regimes, `r=3` factors, seasonal structure,
and multiple observation channels, 12 points leave the dynamics (AR
coefficients, transition matrix, factor loadings, seasonality) hopelessly
under-determined — the posterior would be prior-dominated and degenerate.
The long panel is required to *estimate the dynamics*. The legitimate concern
behind the question — that a stationary AR's "long-run mean" shouldn't be
anchored by a decade of a structurally different, low-volume, pre-CNA-
expansion / pre-AI-tooling era — is better served by:
- **Continuous time-weighting** (a tempered / power likelihood that
  geometrically down-weights older months), so recent structure dominates the
  level *without* discarding the data needed to pin the dynamics;
- a **moderate rolling/expanding window** (e.g. 5-7 years) as a robustness
  check, not 1 year;
- **time-varying parameters / the trend term in (B)**, letting the model
  *learn* the shift instead of averaging across it;
- and recognizing that **`e_t` already absorbs most of the low-frequency
  level growth** — so the "recent regime" intuition is largely handled once
  `e_t` is projected forward correctly (A). In other words, the "use only
  recent data" impulse is mostly a symptom of the §5 `e_t` bug, not an
  independent need to shorten the window.

### 6.4 Suggested minimal path vs. fuller research programme

- *Minimal, do-first:* implement A-minimal (carry `e_t` into `predict()`),
  re-run the §5 backtest, and report the corrected numbers. This is a small,
  clearly-correct change that addresses the dominant error.
- *Fuller:* A-best (latent trend for `e_t` inside the model) + B (drift term
  and/or sticky regimes) + C (dispersion/coverage) + a dedicated
  discovery-vs-realized-risk discussion, and — if the paper wants to make the
  AI claim directly rather than as a caveat — an exogenous AI-capability proxy
  (e.g. an index from major AI-security-tool release dates, counts of
  AI-attributed CVEs, or a post-2023 ramp regressor) tested for incremental
  explanatory power in `e_t` beyond CNA-count growth.

## 7. Implemented fixes (2026-07): forecast covariates + sequential backtest — **update the forecast/evaluation text**

Both changes below are now in the code. They affect numbers the manuscript
reports, so the forecast-methodology and evaluation sections must match.

### 7.1 `predict()` now carries the covariates the fitted likelihood contains

The §6.4 "minimal, do-first" fix is implemented in
`src/cassandra_threatcast/model/full.py::predict()`:

- **Effort offset `e_t`.** The fitted N-channel likelihood is
  `N_kt ~ NegBin2(exp(η_kt + e_t), ψ_k)` (Eq. 4). The forecast now holds
  `e_t` at its last pre-forecast value: `log μ = η + e_T` for all horizons.
  Previously the forecast used `log μ = η` alone, implicitly resetting
  effort to its full-sample mean (e_t is standardized), which under-predicted
  counts by `exp(e_T)` ≈ 3.7× at end-2024 effort levels — the dominant §5
  error. *Manuscript impact:* state the forecast mean as
  `E[N_{k,T+h}] = exp(η_{k,T+h} + e_T)` with a hold-last effort projection
  (or, better, adopt the §6 A-best latent-trend treatment).
- **Baseline disclosure rate `π_s`.** The fitted D-channel rate is
  `ρ_s (Σ_k M_skt λ_kt) + π_s` (additive baseline, one per sector; see §3).
  The forecast previously dropped `π_s`; it is now included.

### 7.2 The backtest is sequential one-step-ahead filtered prediction

`scripts/backtest.py` no longer scores a frozen open-loop trajectory.
For every posterior draw, the Hamilton forward filter runs from the first
panel month (uniform initial regime distribution), and each month t is
predicted from `P(z_t | y_{1:t-1})` plus the factor state at t−1 **before**
that month's data enters the filter; the belief update then uses all four
channels via the in-model `loglik_regime_t`. Prediction-step covariates
(`e_t`, `M_skt`) enter lagged one month. A pure-forecast tail (no further
updates, hold-last covariates) extends past the panel end.

Headline effect (test window 2023-01..2024-12, all-topic panel): 90%
predictive-interval coverage moves from 4.2% under the open-loop §5 design
to ≈98% under sequential evaluation (mildly *under*confident vs. the 90%
target; 50%-interval coverage ≈65% vs. 50%). *Manuscript impact:* this is
the rolling-origin evaluation §4.1 of the paper promises; report coverage
from the sequential design, and present open-loop multi-step forecasts only
as forecasts, not as the calibration evidence.

Honest caveats (documented in the script): static parameters and the
smoothed factor states come from an MCMC fit on the full sample, so the
test window is out-of-sample only with respect to the sequential state
updates; and `e_t` is itself a two-sided HP-filter estimate, so even its
lagged value embeds some future information. A fully out-of-sample variant
retrains on data through 2022 and rebuilds `e_t` with a one-sided estimator
(`build_features.py --effort-method state_space`).

## 8. Paper-conformance completions (2026-07) — **numbers for Tables 2-8 now exist**

The remaining gaps between the manuscript's promises and the implementation
are closed. Each item below changes or fills a paper table.

### 8.1 BSTS-U baseline and logarithmic score (Tables 2, 3, 5, 6)

`run_all_baselines` now fits the univariate BSTS (local linear trend, NUTS)
per series alongside RF/ARIMA/ETS/Naïve, and `evaluate_forecasts` reports
LogS — a moment-matched NegBin log predictive density, negatively oriented
(lower better) — next to CRPS/MAE/RMSE for every model including the
residual-bootstrap baselines.

### 8.2 Damage-function calibration and the economic backtest (Tables 7, 8)

`scripts/calibrate_damage.py` implements the paper's Bayesian calibration of
φ_s against the reference event set (configs/reference_events.yaml documents
each event's loss range and sources). Method: pre-event predictive shock
loads (Hamilton-filtered regime + one-step state propagation, i.e.
"conditioned on pre-event information"), a zero-anchored sigmoid damage
family with **sector-relative operating points** (scale_s = c·median load_s,
shape_s = steepness/median load_s; three global parameters), and NUTS over
the three globals with the documented range as a 90% log-normal interval.

Two consequences the manuscript MUST reflect:
- **The previous fixed parameters (shape 2, scale 1, max 0.5) saturated the
  damage sigmoid for every posterior draw**, collapsing the loss
  distribution to a point mass — VaR = ES = median. All previously
  generated loss figures are invalid.
- **A units error compounded it**: BEA gross output is in $ millions, but
  losses were labeled USD. The corrected pipeline (x_s × 1e6) with
  calibrated parameters yields event-scale losses in the billions —
  consistent with the documented reference events.

The events alone leave max_damage unidentified above what they pin (forecast
months then saturate the sigmoid at an implausible cap), so the calibration
adds an **aggregate annual anchor**: documented estimates of total annual US
cyber losses (CEA 2018 $57-109B; FBI IC3 reported-loss floor; industry
ceiling ~$200B; see configs/reference_events.yaml). Events identify the
event-scale response; the anchor identifies the level.

Table 7 result (anchored run): NotPetya and Change Healthcare covered;
Colonial Pipeline over-predicted ($3.5-6.5B vs documented $0.05-2B) and
MOVEit under-predicted ($2.3-4.3B vs documented $5-15B). Coverage 2/4 with
informative widths — an honest validation finding: the monthly systemic
response discriminates event months only weakly, because individual-event
severity is not fully visible in monthly aggregate threat intensity. The
manuscript should report this as a limitation the framework surfaces (and
scenario studies cannot).

Also produced by `scripts/forecast.py` under the calibrated parameters:
systemic-event probabilities P(ℓ_agg > c) for policy thresholds
(systemic_probs.csv, paper Problem 2), the ranked sector-exposure Table 8
(sector_exposure.csv: exposure index with 90% CI, loss contribution %,
dominant threat topics), and the regime-probability early-warning series
(regime_probs.csv + interactive chart).

### 8.3 Ablation study (Table 4)

`scripts/ablation.py` trains and scores seven settings-matched variants:
full, −regimes (R=1), −factors (Γ=0), −hierarchy (prior scales ×10),
−exploitation channel, −incident channel, −effort (e_t≡0). Model flags live
in `full.py` under `config["ablation"]`; scoring is sequential one-step and
six-step CRPS plus 90% coverage over the 2023-2024 test window
(results/ablation/ablation.csv).

### 8.4 Sparse-topic population (§4.1 evaluation set + Appendix)

`scripts/full_population.py` builds the K=1024 topic population
(MiniBatchNMF over the full CVE corpus), derives the active-topic set with
the paper's exact criteria (≥50 CVE assignments and ≥24 non-zero months in
the initial training window), trains the Full model on the complete
population via VI (the paper's sanctioned scalable alternative for K in the
thousands), and scores Full vs BSTS-U on the full population, the active
set, and the sparse tail (results/full_population/appendix_scores.csv).
Caveats recorded in its meta.json: E/B channels are fully missing at this
granularity (NaN-masked) and the exposure map is uniform.

Related fix: `vi_predictive_samples` passed an empty config into the model,
crashing the VI predictive path; it now threads the config through
(inference/vi.py, scripts/train.py).

## 9. Learned temporal kernel — quantifying "how close is close"

The model's assumption that nearby months predict each other was implicit
(AR(1) factor dynamics imply geometrically decaying correlation Φ^d). The
kernel component makes it explicit, learnable, and testable.

**Construction (moving-average kernel factor).** A smooth-edged square
window of learnable half-width h (months) slides over iid innovations
u_t ~ N(0,1):

    g_t = σ_g · Σ_d w_d(h) · u_{t−d},   w_d(h) = sigmoid((h − |d|)/2),
    weights unit-normalised so Var(g_t) = σ_g².

Each topic loads on the common component: η_kt gains a_k · g_t. The
implied temporal covariance is

    Cov(g_t, g_{t+d}) = σ_g² Σ_j w_j(h) w_{j+d}(h),

nonzero exactly when months t and t+d fall inside overlapping windows —
the "square between 2016 and 2020 centred on the prediction", with the
square's width learned from data rather than assumed. PSD by construction
(a convolution of white noise), and forecastable: beyond the data, future
u's are fresh standard-normal draws; the sequential evaluation conditions
g on innovations through t−h only, replacing newer in-window u's by their
exact Gaussian predictive (variance 1 − Σ w²_known via the unit norm).

**Interpretation of the learned parameters.**
- h    : the temporal reach of local covariance ("how close is close");
- σ_g  : how much local co-movement matters beyond the AR factors,
         regimes, and effort trend — if the data does not support extra
         local covariance, σ_g shrinks toward 0 and the component
         self-ablates;
- a_k  : which topics participate in it.

Priors: h ~ LogNormal(log 24, 0.5) (centred on a ±2-year window),
σ_g ~ HalfNormal(0.5), a_k ~ N(0,1). Config block `kernel:` in
configs/default.yaml (disabled by default — paper-native unchanged).

**Evaluation.** Trained as an ablation-style ADDITION variant
(scripts/ablation.py --variants kernel), settings-matched against the
"full" reference; scored with sequential one-step and six-step CRPS and
90% coverage on the 2023-2024 test window (results/ablation/ablation.csv,
"kernel" row). Manuscript impact if adopted: Eq. (3) gains the term
a_k g_t with the kernel definition above, and Table 4 gains a "+ kernel"
row quantifying the improvement.
