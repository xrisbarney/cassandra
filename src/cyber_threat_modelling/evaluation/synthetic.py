"""Independent NumPy generator for Stage 7 fixed-truth recovery experiments."""

import numpy as np
from scipy.special import expit

from ..models.components import kernel_weights

SCENARIOS = (
    "independent", "latent_ar", "factors_sparse", "severity_missing", "regimes",
    "effort_proxy", "effort_oracle", "effort_none", "kernel_present", "kernel_absent",
    "dispersion_dense", "dispersion_sparse", "misspecified", "representative",
)


def stationary_factors(rng, phi, months):
    phi = np.asarray(phi)
    f = np.empty((months + 1, len(phi)))
    f[0] = rng.normal(size=len(phi)) / np.sqrt(1 - phi**2)
    for t in range(1, months + 1):
        f[t] = phi * f[t - 1] + rng.normal(size=len(phi))
    return f


def generate(scenario, seed):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown synthetic scenario.")
    # The effort variants deliberately use exactly the same observations.
    rng = np.random.default_rng(seed)
    k, t, rank = (10, 108, 2) if scenario == "representative" else (4, 96, 1)
    if scenario.startswith("kernel"):
        t = 144
    residual = "none"
    if scenario == "independent":
        rank = 0
    if scenario == "latent_ar" or scenario.startswith("dispersion"):
        k, rank = 1, 0
        residual = "ar" if scenario == "latent_ar" else "iid"
    if scenario.startswith("dispersion"):
        t = 192
    severity = scenario == "severity_missing"
    regimes = 2 if scenario == "regimes" else 1
    kernel = scenario.startswith("kernel")
    n = t + 12
    m = np.log(np.geomspace(100, 1, k))
    if scenario in ('factors_sparse', 'representative'):
        # A genuinely rare last category: approximately ten publications in a
        # 96-month window, rather than merely a smaller but regularly nonzero series.
        m[-1] = np.log(0.1)
    if k == 1:
        m = np.array([np.log(0.8 if scenario == "dispersion_sparse" else 40)])
    phi = np.array([0.65, 0.25])[:rank]
    loading = np.empty((k, rank))
    for j in range(rank):
        loading[:, j] = np.linspace(0.45, -0.25, k)
        loading[:j, j] = 0
        loading[j, j] = 0.45
    f = stationary_factors(rng, phi, n)
    tau = np.full(k, 0.35)
    u = np.zeros((n, k))
    phi_u = np.full(k, 0.55)
    if residual == "iid":
        u = rng.normal(size=(n, k)) * tau
    elif residual == "ar":
        u = stationary_factors(rng, phi_u, n)[1:] * tau
    a = np.zeros(k)
    if scenario == "kernel_present":
        a = np.linspace(0.65, 0.25, k)
    width = 8.0
    xi = rng.normal(size=n + 60)
    g = np.convolve(xi, kernel_weights(width), mode="valid")
    transition = np.array([[0.94, 0.06], [0.06, 0.94]]) if regimes == 2 else np.ones((1, 1))
    z = np.zeros(n, dtype=int)
    if regimes == 2:
        # Fixed-truth transition stress case, conditional on this declared path.
        z[t // 3: 2 * t // 3] = 1
        z[t - 8:] = 1
    shifts = np.array([0, 1.4]) if regimes == 2 else np.array([0.0])
    eta = m + f[1:] @ loading.T + u + g[:, None] * a + shifts[z, None]
    true_effort = np.zeros(n)
    if scenario.startswith("effort"):
        true_effort[t // 2:] = 1.2
    psi = np.full(k, 12.0)
    mu = np.exp(eta + true_effort[:, None])
    counts = rng.negative_binomial(psi, psi / (psi + mu))
    if scenario == "misspecified":
        # Unmodelled selection/zero inflation, deliberately outside the fitted NB law.
        counts[rng.random(counts.shape) < 0.3] = 0
    mask_n = np.ones((t, k), dtype=bool)
    if severity:
        mask_n[24:28] = False
    offset = np.zeros(t)
    if scenario == "effort_oracle":
        offset = true_effort[:t].copy()
    data = {"N": np.where(mask_n, counts[:t], np.nan), "mask_N": mask_n, "effort": offset}
    truth = {
        "m": m, "phi": phi, "loading": loading, "f": f[:t + 1],
        "tau": tau, "phi_u": phi_u, "u": u[:t], "psi": psi, "a": a,
        "width": np.array(width), "g": g[:t], "eta": eta[:t], "mu": mu[:t],
        "regime": z[:t], "shift": shifts, "transition": transition,
        "true_effort": true_effort[:t], "future_counts": counts[t:],
        "future_eta": eta[t:], "future_mu": mu[t:],
    }
    if severity:
        phi_h = np.array([0.45])
        h = stationary_factors(rng, phi_h, n)
        loading_h = np.array([[0.35], [0.25], [-0.2], [0.15]])
        location = np.linspace(1.0, 0.1, k) + h[1:] @ loading_h.T
        kappa = np.full(k, 0.3)
        b = np.clip(10 * expit(location + rng.normal(size=(n, k)) * kappa), 0.05, 9.95)
        e = np.clip(expit(-3 + 0.4 * eta + rng.normal(size=(n, k)) * 0.5), 1e-4, 1-1e-4)
        mu_v = np.exp(-3 + 0.5 * eta)
        v = rng.negative_binomial(2, 2 / (2 + mu_v))
        exposure = np.full((2, k), 0.25)
        rate = 0.02 * (np.exp(eta) @ exposure.T) + 0.1
        incidents = rng.poisson(rate)
        for name, values in [("B", b), ("E", e), ("V", v), ("D", incidents)]:
            mask = rng.random(values[:t].shape) > 0.25
            mask[:12] = False
            if name in ("B", "E"):
                mask &= (counts[:t] > 0) & mask_n
            data[name] = np.where(mask, values[:t], np.nan)
            data[f"mask_{name}"] = mask
        data["M"] = exposure
        truth.update({
            "severity_location": location[:t], "sigma": expit(location[:t]),
            "n": np.linspace(1.0, 0.1, k), "kappa": kappa,
            "loading_h": loading_h, "phi_h": phi_h,
        })
    options = {
        "scenario": scenario, "seed": seed, "months": t, "categories": k,
        "factors": rank, "severity_factors": 1 if severity else 0,
        "regimes": regimes, "residual": residual, "kernel": kernel,
        "conditional_level": scenario.startswith("dispersion"),
        "known_auxiliary_coefficients": severity,
    }
    return data, truth, options
