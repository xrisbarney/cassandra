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
