"""Paper-conformant model, regime recovery, and forecast (manuscript §3.2–3.6).

The manuscript specifies, verbatim:

§3.2  f_t = Φ f_{t-1} + w_t, w ~ N(0, Q), dynamics SHARED BY ALL REGIMES
      ("a Markov-switching mean rather than a full Markov-switching VAR");
      η_t = μ_{z_t} + Γ f_t + a g_t + u_t, where g_t is the moving-window
      kernel (Eqs. 5–6) — §4.3 marks the kernel OPTIONAL ("the final row
      adds the optional moving-window kernel of Section 3 to the full
      model"), so it is governed by config `kernel.enabled` (off in the
      headline Full model, on in the "+ moving-window kernel" ablation);
      severities analogous with ν_{z_t}, Ψ, h_t; leading r×r blocks of Γ
      and Ψ lower triangular with strictly positive diagonal; half-Normal
      scales floored away from zero; standard normal free loadings.
§3.3  N ~ NegBin(e_t λ, ψ) with e_t the standardized smooth trend covariate;
      logit E ~ N(α + β log λ, ς²); D ~ Poisson(ρ_s Σ M λ + π_s) with π_s
      constant and additive; B ~ LogNormal(log σ, κ²); E/B masked when
      missing; channels conditionally independent given the latent state.
§3.5  The regime path is marginalized analytically inside the likelihood by
      the HMM forward recursion; NUTS samples every continuous unknown; no
      categorical variable is ever sampled; regime paths are recovered
      afterwards by forward-filtering backward-sampling.
§3.6  Forecast: z from Π, f via Eq. 3, fresh kernel innovations at the
      posterior width, e_t held at its last observed value, then Eqs. 1–2.

All of that is implemented by `model.full.full_model` and
`model.full.predict`; this module pins the paper configuration and adds the
§3.5 post-hoc regime recovery.
"""
from __future__ import annotations

import copy

import numpy as np

from cassandra_threatcast.inference.ffbs import ffbs
from cassandra_threatcast.model import full as _full

predict = _full.predict  # §3.6 forecast recursion (e_t held at last value)


def paper_config(config: dict) -> dict:
    """Return a copy of `config` pinned to the manuscript's model.

    No Hawkes term exists in the stated model (§3.2 lists a Hawkes intensity
    only as "a natural alternative"), and innovations are Gaussian (§3.2).
    The kernel and the Table-4 ablation switches pass through unchanged:
    §4.3 defines the kernel as optional and the ablation variants as part of
    the evaluation.
    """
    cfg = copy.deepcopy(config)
    cfg.pop("hawkes", None)
    cfg.setdefault("enhanced", {})["enabled"] = False
    return cfg


def paper_model(data: dict, config: dict) -> None:
    """§3.2–3.3 model with the regime marginalized in-likelihood (§3.5)."""
    _full.full_model(data, paper_config(config))


def recover_regime_paths(posterior: dict, seed: int = 0) -> np.ndarray:
    """§3.5: recover posterior regime paths by FFBS, one path per draw.

    posterior : flattened draws containing `loglik_regime_t` (n, T, R) —
                the per-regime channel log-likelihoods the forward recursion
                evaluated during sampling — and `Pi` (n, R, R).
    Returns z_t of shape (n, T), int regime labels.
    """
    loglik = np.asarray(posterior["loglik_regime_t"], dtype=float)
    Pi = np.asarray(posterior["Pi"], dtype=float)
    n, T, R = loglik.shape
    pi0 = np.ones(R) / R  # uniform initial regime, matching the model
    z_paths = np.zeros((n, T), dtype=int)
    for i in range(n):
        z_paths[i] = ffbs(loglik[i], Pi[i], pi0, n_samples=1, seed=seed + i)[0]
    return z_paths
