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

- **Non-centered idiosyncratic noise.** `u_t` and `v_t` are sampled as
  `standardized ε ~ N(0,1)` then scaled (`u = τ·ε`), instead of directly from
  `N(0, τ²)`. Removes Neal's-funnel geometry. Mathematically identical model.
- **Stationary AR.** The factor/severity AR matrices are passed through `tanh`
  before the lower-triangular mask, bounding eigenvalues to (−1, 1) so factors
  cannot explode over the ~180-month series.
- **Overflow clamps.** Log-intensities are clipped to [−30, 30] before `exp` to
  keep the Negative-Binomial mean finite.
- **NaN masking.** Topic-months with no CVEs carry NaN in the EPSS and severity
  channels; those are masked out of their likelihoods.

## Result

With items 1–3 fixed, the adapted NUTS step size goes from ~`3.6e-10` (no
sampling) to ~`1e-3`–`1e-2` (healthy) on the real 180-month panel.
