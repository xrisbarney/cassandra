"""Stage 6 Bayesian controls with frozen laws and explicit diagnostic gates."""

import time
import warnings

import numpy as np

from ..data.store import write_json
from ..forecasting.count_laws import CountForecast, integer_counts
from .count_baselines import FitFailure, stream_seed


def pooled_prior_covariance(categories, center):
    """Covariance of (b0+b[k]+beta*center, beta) under the frozen independent priors."""
    cov = np.empty((categories + 1, categories + 1))
    cov[:categories, :categories] = np.eye(categories) + (4 + 0.25 * center**2)
    cov[:categories, categories] = 0.25 * center
    cov[categories, :categories] = 0.25 * center
    cov[categories, categories] = 0.25
    return cov


def build_model(train, initial_counts, method):
    import pymc as pm
    import pytensor.tensor as pt

    y = integer_counts(train)
    initial = integer_counts(initial_counts)
    if y.ndim != 2 or initial.ndim != 2 or initial.shape[1] != y.shape[1]:
        raise ValueError("Incompatible training and initial-prior anchor arrays.")
    t, k = y.shape
    if method == "independent_nb_ar1":
        from .independent_geometry import build_conditioned_model

        # Frozen Stage 5 row order: category 8 is the strongly informed persistent
        # series. Other states use innovations. This is an exact coordinate choice
        # from initial-window diagnostics, not forecast-score tuning.
        centered = (
            np.array([7], dtype=int) if k > 7 and initial[:, 7].sum() else np.array([], dtype=int)
        )
        return build_conditioned_model(y, initial, center_states=centered)
    with pm.Model(coords={"category": np.arange(k), "month": np.arange(t)}) as model:
        log_psi = pm.Normal("log_psi", 0, 1, dims="category")
        psi = pm.Deterministic("psi", pt.exp(log_psi), dims="category")
        if method == "pooled_nb":
            # Exact Gaussian reparameterization: eliminate the b0/b ridge from
            # the likelihood and center time, without changing any prior/law.
            center = (t - 1) / 24
            prior_mean = np.log(initial.mean() + 0.5)
            coefficients = pm.MvNormal(
                "centered_coefficients",
                mu=np.r_[np.full(k, prior_mean), 0.0],
                cov=pooled_prior_covariance(k, center),
                shape=k + 1,
            )
            beta = pm.Deterministic("beta", coefficients[-1])
            alpha = coefficients[:k] - beta * center
            b0_residual = pm.Normal("b0_residual", 0, 1)
            b0 = pm.Deterministic(
                "b0",
                prior_mean
                + 4 / (1 + 4 * k) * pt.sum(alpha - prior_mean)
                + np.sqrt(4 / (1 + 4 * k)) * b0_residual,
            )
            pm.Deterministic("b", alpha - b0, dims="category")
            eta = coefficients[:k, None] + beta * (np.arange(t)[None, :] / 12 - center)
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
    stats = idata.sample_stats
    if "tree_depth" in stats:
        saturated = stats.tree_depth >= depth
    elif "maxdepth_reached" in stats:
        saturated = stats.maxdepth_reached
    else:
        raise ValueError("Sampler output lacks a maximum-tree-depth diagnostic.")
    report = {
        "max_rhat": float(summary.r_hat.max()),
        "min_bulk_ess": float(summary.ess_bulk.min()),
        "min_tail_ess": float(summary.ess_tail.min()),
        "divergences": int(idata.sample_stats.diverging.sum()),
        "tree_depth_saturations": int(saturated.sum()),
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
            step = pm.NUTS(
                target_accept=accept, max_treedepth=depth, compile_kwargs={"mode": "NUMBA"}
            )
            compile_seconds = time.perf_counter() - start
            start = time.perf_counter()
            chain_timing = {}
            point = model.initial_point()
            initial_points = []
            for chain in range(4):
                rng = np.random.default_rng(stream_seed(seed, name, origin, retry, chain, "init"))
                initial_points.append(
                    {
                        key: value + rng.normal(0, 0.1, size=value.shape)
                        for key, value in point.items()
                    }
                )

            def progress(trace, draw):
                now = time.perf_counter() - start
                record = chain_timing.setdefault(str(draw.chain), {"first_draw_seconds": now})
                if not draw.tuning and "first_retained_draw_seconds" not in record:
                    record["first_retained_draw_seconds"] = now
                if draw.is_last:
                    record["last_draw_seconds"] = now
                if draw.chain == 0 and (draw.draw_idx + 1) % 500 == 0:
                    print(
                        f"{name} {origin} retry {retry}: chain 0 draw {draw.draw_idx + 1}",
                        flush=True,
                    )

            idata = pm.sample(
                draws=retained,
                tune=warmup,
                chains=4,
                cores=1,
                step=step,
                random_seed=stream_seed(seed, name, origin, retry, "posterior"),
                progressbar=False,
                callback=progress,
                initvals=initial_points,
            )
            sample_seconds = time.perf_counter() - start
        diag, table = diagnostic_report(idata, depth)
        attempt = {
            "retry_number": retry,
            "warmup": warmup,
            "draws": retained,
            "chains": 4,
            "cores": 1,
            "backend": "PyMC NUTS with PyTensor NUMBA log-density/gradient",
            "initialization": "distinct seeded 0.1-SD jitter in unconstrained coordinates",
            "target_accept": accept,
            "max_treedepth": depth,
            "graph_seconds": construction,
            "step_construction_compile_seconds": compile_seconds,
            "sample_call_including_startup_and_warmup_seconds": sample_seconds,
            "chain_wall_markers": chain_timing,
            "warnings": [str(w.message) for w in caught],
            "diagnostics": diag,
        }
        # A documented missing-BLAS performance warning does not invalidate NUTS
        # arithmetic. Preserve it; reject every unresolved numerical warning.
        environment = [
            str(w.message)
            for w in caught
            if str(w.message).startswith("PyTensor could not link to a BLAS installation.")
        ]
        unresolved = [str(w.message) for w in caught if str(w.message) not in environment]
        attempt["environment_warnings"] = environment
        attempt["unresolved_warnings"] = unresolved
        attempt["fit_status"] = "accepted" if diag["accepted"] and not unresolved else "failed"
        serialization_start = time.perf_counter()
        idata.to_netcdf(folder / f"posterior_attempt_{retry}.nc")
        attempt["posterior_serialization_seconds"] = time.perf_counter() - serialization_start
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
            from ..evaluation.control_checks import predictive_checks

            checks = predictive_checks(idata.posterior, name, train, initial_counts, origin)
            write_json(folder / "predictive_checks.json", checks)
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
