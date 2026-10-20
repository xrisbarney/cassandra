"""Version A and pooled NB controls; substantive use awaits Stage 7 validation."""

import time
import warnings

import numpy as np

from ..data.store import write_json
from ..forecasting.count_laws import CountForecast, integer_counts
from .count_baselines import FitFailure, stream_seed


def build_model(train, initial_counts, method):
    import pymc as pm
    import pytensor.tensor as pt

    y = integer_counts(train)
    initial = integer_counts(initial_counts)
    if y.ndim != 2 or initial.ndim != 2 or initial.shape[1] != y.shape[1]:
        raise ValueError("Incompatible training and initial-prior anchor arrays.")
    t, k = y.shape
    with pm.Model(coords={"category": np.arange(k), "month": np.arange(t)}) as model:
        log_psi = pm.Normal("log_psi", 0, 1, dims="category")
        psi = pm.Deterministic("psi", pt.exp(log_psi), dims="category")
        if method == "independent_nb_ar1":
            m = pm.Normal("m", np.log(initial.mean(axis=0) + 0.5), 2, dims="category")
            tau = pm.HalfNormal("tau", 0.3, dims="category")
            phi = pm.Uniform("phi", -1, 1, dims="category")
            unit = pm.AR(
                "unit_state",
                rho=phi[:, None],
                sigma=1,
                init_dist=pm.Normal.dist(0, 1 / pt.sqrt(1 - phi[:, None] ** 2), shape=(k, 1)),
                ar_order=1,
                shape=(k, t),
                dims=("category", "month"),
            )
            eta = m[:, None] + tau[:, None] * unit
            pm.Deterministic("terminal_u", tau * unit[:, -1], dims="category")
        elif method == "pooled_nb":
            b0 = pm.Normal("b0", np.log(initial.mean() + 0.5), 2)
            b = pm.Normal("b", 0, 1, dims="category")
            beta = pm.Normal("beta", 0, 0.5)
            eta = b0 + b[:, None] + beta * np.arange(t)[None, :] / 12
        else:
            raise ValueError("Unknown Bayesian control.")
        pm.NegativeBinomial(
            "counts", mu=pt.exp(eta), alpha=psi[:, None], observed=y.T, dims=("category", "month")
        )
    return model


def diagnostic_report(idata, depth):
    import arviz as az

    summary = az.summary(idata, kind="diagnostics", round_to="none")
    finite = np.isfinite(summary[["r_hat", "ess_bulk", "ess_tail"]]).all().all()
    report = {
        "max_rhat": float(summary.r_hat.max()),
        "min_bulk_ess": float(summary.ess_bulk.min()),
        "min_tail_ess": float(summary.ess_tail.min()),
        "divergences": int(idata.sample_stats.diverging.sum()),
        "tree_depth_saturations": int((idata.sample_stats.tree_depth >= depth).sum()),
        "min_bfmi": float(np.min(az.bfmi(idata))),
        "all_diagnostics_finite": bool(finite),
    }
    report["accepted"] = bool(
        finite
        and report["max_rhat"] < 1.01
        and report["min_bulk_ess"] > 400
        and report["min_tail_ess"] > 400
        and report["divergences"] == 0
        and report["tree_depth_saturations"] == 0
        and report["min_bfmi"] > 0.3
    )
    return report, summary


def posterior_forecast(posterior, method, months, draws, seed):
    """Preserve common coefficient draws for the pooled model and whole latent paths."""
    post = posterior.stack(sample=("chain", "draw"))
    n = post.sizes["sample"]
    k = post.sizes["category"]
    rng = np.random.default_rng(seed)
    shared_indices = rng.integers(n, size=draws)
    means, concentrations = np.empty((draws, 12, k)), np.empty((draws, 12, k))
    for j in range(k):
        # Independent controls have independent parameter streams across categories.
        indices = rng.integers(n, size=draws) if method == "independent_nb_ar1" else shared_indices
        psi = post.psi.isel(category=j).values[indices]
        if method == "independent_nb_ar1":
            phi = post.phi.isel(category=j).values[indices]
            tau = post.tau.isel(category=j).values[indices]
            m = post.m.isel(category=j).values[indices]
            u = post.terminal_u.isel(category=j).values[indices].copy()
        else:
            b0 = post.b0.values[indices]
            b = post.b.isel(category=j).values[indices]
            beta = post.beta.values[indices]
        for h in range(12):
            if method == "independent_nb_ar1":
                u = phi * u + tau * rng.normal(size=draws)
                eta = m + u
            else:
                eta = b0 + b + beta * (months + h) / 12
            with np.errstate(over="raise", under="raise"):
                means[:, h, j] = np.exp(eta)
            concentrations[:, h, j] = psi
    paths = rng.negative_binomial(concentrations, concentrations / (concentrations + means))
    return CountForecast(
        "nb2_mixture", paths, {"mean": means, "concentration": concentrations}
    ).validate()


def fit_predict_bayesian(train, initial_counts, spec, draws, origin, folder, seed=20261020):
    import pymc as pm
    import pytensor

    if not pytensor.config.cxx:
        raise ValueError("Activate the verified compiler before Bayesian fitting.")
    name = spec["method"]
    started = time.perf_counter()
    model = build_model(train, initial_counts, name)
    construction = time.perf_counter() - started
    attempts = []
    budgets = ((1000, 1000, 0.95, 12), (2000, 1000, 0.99, 15), (2000, 2000, 0.99, 15))
    for retry, (warmup, retained, accept, depth) in enumerate(budgets):
        with model, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            step = pm.NUTS(target_accept=accept, max_treedepth=depth)
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
            idata = pm.sample(
                draws=retained,
                tune=warmup,
                chains=4,
                cores=4,
                step=step,
                random_seed=stream_seed(seed, name, origin, retry, "posterior"),
                progressbar=False,
            )
            sample_seconds = time.perf_counter() - start
        diag, table = diagnostic_report(idata, depth)
        attempt = {
            "retry_number": retry,
            "warmup": warmup,
            "draws": retained,
            "chains": 4,
            "target_accept": accept,
            "max_treedepth": depth,
            "graph_seconds": construction,
            "step_construction_compile_seconds": compile_seconds,
            "sample_call_including_startup_and_warmup_seconds": sample_seconds,
            "warnings": [str(w.message) for w in caught],
            "diagnostics": diag,
        }
        # Unknown warnings require inspection; no fit is accepted just for exiting.
        attempt["fit_status"] = "accepted" if diag["accepted"] and not caught else "failed"
        idata.to_netcdf(folder / f"posterior_attempt_{retry}.nc")
        table.to_csv(folder / f"diagnostics_attempt_{retry}.csv")
        # JSON explicitly represents undefined diagnostics as null, not fabricated values.
        safe = {
            **attempt,
            "diagnostics": {
                k: v if not isinstance(v, float) or np.isfinite(v) else None
                for k, v in diag.items()
            },
        }
        write_json(folder / f"attempt_{retry}.json", safe)
        attempts.append(safe)
        if attempt["fit_status"] == "accepted":
            future = posterior_forecast(
                idata.posterior, name, len(train), draws, stream_seed(seed, name, origin, "future")
            )
            return future, {
                "fit_status": "accepted",
                "retry_number": retry,
                "attempts": attempts,
                "parameter_uncertainty": "posterior draws",
                "cross_category_dependence": "shared coefficients"
                if name == "pooled_nb"
                else "independent",
            }
    raise FitFailure("Bayesian control failed the diagnostic/retry gate.", attempts)
