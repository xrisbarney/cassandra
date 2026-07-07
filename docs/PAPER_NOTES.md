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

## Result and a residual caveat

With items 1–3 fixed, the adapted NUTS step size goes from ~`3.6e-10` (no
sampling at all) to ~`1e-3`–`1e-2` (healthy) on the real 180-month panel, and
out-of-sample calibration is good (empirical coverage ≈ nominal coverage;
PIT not significantly non-uniform) once a run adapts successfully.

**However, this specific model + real-data combination — NUTS wrapped in
`DiscreteHMCGibbs` for the discrete regime path, on the actual 180-month,
K=8/S=11/R=3 panel — does not adapt successfully for every random seed.**
Empirically, a meaningful fraction of seeds cause the warmup step-size
adaptation to collapse to numerical underflow (~1e-38) at some point during
warmup, seemingly as a property of that seed's specific HMC trajectory rather
than a further fixable structural defect: it happens unpredictably across
different seeds, warmup lengths, and with every one of the mitigations above
individually applied (identifiability, non-centering, scale floors, smooth
saturation, informed initialization) — none eliminates it outright, though
together they made the *healthy* outcome (the ~1e-3 step size / good
calibration case above) achievable at all, which it was not before item 1–3
were fixed.

**Mitigation:** `run_nuts` (in `src/cassandra_threatcast/inference/nuts.py`)
automatically detects a collapsed step size after each attempt and retries
with a new seed, up to `_MAX_ADAPTATION_RETRIES` times (default 6). This is a
pragmatic safety net, not a proof that the underlying difficulty is resolved
— on the real 180-month panel, the failure rate per attempt has been high
enough in testing (many consecutive seeds collapsing) that even 6 retries are
not a strong guarantee. If you see the "NUTS adaptation did not recover"
warning after all retries, the returned posterior should not be trusted.

**Recommended long-term fix (not yet implemented): marginalize the discrete
regime instead of Gibbs-sampling it.** `numpyro`'s own documentation flags
`DiscreteHMCGibbs` as an `[EXPERIMENTAL INTERFACE]` and explicitly recommends
its alternative for exactly this situation: enumerate/marginalize the
discrete latent (`z_t`) analytically via `infer={"enumerate": "parallel"}`
(NumPyro's funsor-based enumeration, as used in its own HMM examples) instead
of resampling it with Gibbs steps interleaved with NUTS. This would let plain
NUTS — mature, well-tested, no interleaved-kernel target-shifting — handle
the entire continuous parameter space, likely eliminating this failure mode
outright rather than working around it with retries. This is a nontrivial
rewrite of the regime-switching part of `full_model` (replacing the discrete
`numpyro.sample("z_t", ...)` inside the `scan` with a funsor-compatible
enumerated formulation) and was out of scope to implement and validate in the
time available here, but is the principled next step if retry-based
mitigation proves insufficient in practice.
