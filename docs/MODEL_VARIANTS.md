# Model variants: paper-native vs. enhanced

The codebase runs two model variants from the same code path:

- **Paper-native** (default): the model exactly as specified in the manuscript.
  No flag needed — `python scripts/train.py` runs this.
- **Enhanced** (`--enhanced-mode`): a set of Claude-proposed statistical
  enhancements that go *beyond* the paper, aimed at the tail-risk quantities
  the paper itself cares about (predictive VaR / ES, systemic-event
  probabilities). Enable with `python scripts/train.py --enhanced-mode`
  (and pass `--enhanced-mode` to `forecast.py` too so the forward simulation
  matches how the model was fit).

The switch is read from `config["enhanced"]["enabled"]` (see
`configs/default.yaml`); the CLI flag just sets it to `true`.

---

## Enhancements behind `--enhanced-mode` (implemented)

| Component | Paper-native model | Claude's enhanced proposal | Potential improvement |
|-----------|--------------------|----------------------------|-----------------------|
| **Factor & severity innovations** — `w_t`, `η` idiosyncratic noise, severity `ξ_t`, `v_t` | Gaussian: `w_t ~ N(0, Q)`, etc. (Eq. 3) | Student-t with `ν = student_t_df` (default 4): `w_t ~ t_ν(0, Q)` | Cyber activity is **bursty and fat-tailed** (Log4Shell-style spikes). Gaussian innovations understate the probability of extreme months, so the paper's `VaR_0.95` / `ES_0.95` are biased low. Heavy tails give honest tail mass without inflating the bulk. |
| **Incident-disclosure channel** — `D_{s,t}` (SEC 8-K counts) | Poisson: `D_{s,t} ~ Poisson(rate)` (Eq. 8), which forces `Var = Mean` | Negative Binomial: `D_{s,t} ~ NegBin2(rate, φ_D)` with a per-sector dispersion `φ_D` | Mandatory 8-K disclosures **cluster** (a single campaign triggers many filings at once), so counts are overdispersed. Poisson under-covers; NegBin restores calibrated predictive intervals and PIT uniformity for the incident channel. |

Both are single, well-scoped changes to the generative model; the priors,
latent structure, regime switching, exposure map, and economic propagation are
identical across variants, so a paper-native vs. enhanced comparison is a clean
ablation. Under `--enhanced-mode` the model gains one extra parameter block
(`φ_D`, per sector) and the innovation distributions change family; everything
else — and all downstream outputs — is unchanged.

## Already applied (not a toggle — correctness alignment)

| Component | Original code | Current code | Why |
|-----------|---------------|--------------|-----|
| **Damage function** `φ_s` | Raw sigmoid `d̄·σ(β(ℓ−c))`, which returns `≈4%` disruption at zero shock load | Anchored sigmoid `d̄·(σ(β(ℓ−c)) − σ(−βc)) / (1 − σ(−βc))`, so `φ_s(0)=0` | The shock load `ℓ_{s,t}=Σ_k M λ σ ≥ 0`; zero load means *no threat activity* and must map to zero disruption. The raw form also biased the single-sector reference-event calibration (NotPetya etc.) upward. This aligns the code with the paper's construction rather than changing it, so it is on in both variants. |

## Documented, not yet implemented (candidate future enhancements)

| Component | Paper-native model | Proposed enhancement | Potential improvement |
|-----------|--------------------|----------------------|-----------------------|
| **Factor innovation contagion** | Independent Gaussian innovation `u_t` (Eq. 3) | Multivariate **Hawkes-type self-exciting** intensity (the paper flags this as an optional extension) | Captures short-horizon cross-topic contagion (one attack class triggering another), matching the self-/cross-excitation Bessy-Roland document. |
| **Reporting propensity** `π_{s,t}` | Static per-cell `HalfNormal` prior | Latent **random walk** `logit π_{s,t} = logit π_{s,t-1} + ζ_t` | Disclosure propensity trends over time (the Item 1.05 mandate is recent); a random walk borrows strength across months instead of estimating each cell independently. |

These are left as documented proposals rather than silent code so the paper's
scope stays explicit; they can be added behind the same `enhanced` switch if
wanted.
