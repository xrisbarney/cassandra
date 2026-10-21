"""Independent category posteriors with reusable code and fresh adaptation."""

import time
import warnings
from importlib.metadata import version

import numpy as np

from ..data.store import write_json
from ..forecasting.count_laws import CountForecast, integer_counts
from .control_priors import INDEPENDENT_PRIOR_RULE, INDEPENDENT_STATIONARY_SCALE_SD
from .count_baselines import FitFailure, stream_seed
from .independent_geometry import build_conditioned_model

BUDGETS = ((1000, 1000, 0.95, 12), (2000, 1000, 0.99, 15), (2000, 2000, 0.99, 15))


def combine_forecasts(forecasts):
    if not forecasts or any(f.family != "nb2_mixture" or f.paths.shape[2] != 1 for f in forecasts):
        raise ValueError("Expected one NB-mixture forecast per category.")
    return CountForecast(
        "nb2_mixture",
        np.concatenate([f.paths for f in forecasts], axis=2),
        {
            key: np.concatenate([f.parameters[key] for f in forecasts], axis=2)
            for key in ("mean", "concentration")
        },
    ).validate()


def forecast_from_files(folder, fit, months, draws, seed):
    import arviz as az

    from .bayesian_controls import posterior_forecast

    forecasts = []
    if [c["category"] for c in fit["per_category"]] != list(range(1, len(fit["per_category"]) + 1)):
        raise ValueError("Independent posterior categories are incomplete or reordered.")
    for entry in fit["per_category"]:
        path = (folder / entry["posterior_file"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Posterior path escapes the run.")
        idata = az.from_netcdf(path)
        forecasts.append(
            posterior_forecast(
                idata.posterior,
                "independent_nb_ar1",
                months,
                draws,
                stream_seed(seed, "category", entry["category"]),
            )
        )
        idata.close()
    return combine_forecasts(forecasts)


def fit_independent_categories(train, initial_counts, draws, origin, folder, seed):
    import nutpie
    import pymc as pm

    from ..evaluation.control_checks import predictive_checks
    from .bayesian_controls import diagnostic_report, posterior_forecast

    if version("nutpie") != "0.15.2":
        raise ValueError("Install the pinned supplemental NUTS runtime.")
    y, initial = integer_counts(train), integer_counts(initial_counts)
    if y.ndim != 2 or initial.ndim != 2 or y.shape[1] != initial.shape[1]:
        raise ValueError("Incompatible independent-control arrays.")
    method = "independent_nb_ar1"
    compiled_graphs, attempts, categories, forecasts, checks = {}, [], [], [], []
    future_seed = stream_seed(seed, method, origin, "future")
    for j in range(y.shape[1]):
        category = j + 1
        subfolder = folder / f"category_{category:02d}"
        subfolder.mkdir(exist_ok=False)
        centered = j == 7
        data = {
            "training_counts": y[:, j : j + 1].T,
            "prior_mean": np.log(initial[:, j : j + 1].mean(axis=0) + 0.5),
        }
        for retry, (warmup, retained, accept, depth) in enumerate(BUDGETS):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                if centered not in compiled_graphs:
                    model = build_conditioned_model(
                        y[:, j : j + 1],
                        initial[:, j : j + 1],
                        center_states=centered,
                        mutable=True,
                        stationary_scale_sd=INDEPENDENT_STATIONARY_SCALE_SD,
                    )
                    compiled = nutpie.compile_pymc_model(model, backend="numba")
                    compiled_graphs[centered] = (model, compiled)
                else:
                    model, compiled = compiled_graphs[centered]
                # Updating observations and the frozen initial-window prior anchor
                # changes data only. Every sampler below is newly constructed.
                pm.set_data(data, model=model)
                compiled = compiled.with_data(**data)
                compile_seconds = time.perf_counter() - start
                start = time.perf_counter()
                idata = nutpie.sample(
                    compiled,
                    draws=retained,
                    tune=warmup,
                    chains=4,
                    cores=1,
                    seed=stream_seed(seed, method, origin, category, retry, "posterior"),
                    target_accept=accept,
                    maxdepth=depth,
                    progress_bar=False,
                    save_warmup=True,
                )
                sample_seconds = time.perf_counter() - start
            diag, table = diagnostic_report(idata, depth)
            messages = [str(w.message) for w in caught]
            unresolved = [
                m
                for m in messages
                if not m.startswith("PyTensor could not link to a BLAS installation.")
            ]
            accepted = diag["accepted"] and not unresolved
            posterior_file = subfolder / f"posterior_attempt_{retry}.nc"
            start = time.perf_counter()
            idata.to_netcdf(posterior_file)
            attempt = {
                "category": category,
                "retry_number": retry,
                "warmup": warmup,
                "draws": retained,
                "chains": 4,
                "cores": 1,
                "target_accept": accept,
                "max_treedepth": depth,
                "backend": "nutpie 0.15.2, NUMBA, gradient diagonal adaptation",
                "state_coordinates": "centered" if centered else "innovations",
                "initial_level": "exact Gaussian conditioning",
                "prior_rule": INDEPENDENT_PRIOR_RULE,
                "stationary_scale_prior_sd": INDEPENDENT_STATIONARY_SCALE_SD,
                "seed": stream_seed(seed, method, origin, category, retry, "posterior"),
                "step_construction_compile_seconds": compile_seconds,
                "sample_call_including_startup_and_warmup_seconds": sample_seconds,
                "posterior_serialization_seconds": time.perf_counter() - start,
                "warnings": messages,
                "unresolved_warnings": unresolved,
                "diagnostics": {
                    key: value if not isinstance(value, float) or np.isfinite(value) else None
                    for key, value in diag.items()
                },
                "fit_status": "accepted" if accepted else "failed",
            }
            table.to_csv(subfolder / f"diagnostics_attempt_{retry}.csv")
            write_json(subfolder / f"attempt_{retry}.json", attempt)
            attempts.append(attempt)
            print(
                f"{method} {origin} category {category} retry {retry}: {attempt['fit_status']}",
                flush=True,
            )
            if accepted:
                check = predictive_checks(
                    idata.posterior,
                    method,
                    y[:, j : j + 1],
                    initial[:, j : j + 1],
                    origin,
                    independent_prior_scale=INDEPENDENT_STATIONARY_SCALE_SD,
                )
                check["category"] = category
                checks.append(check)
                forecasts.append(
                    posterior_forecast(
                        idata.posterior,
                        method,
                        len(y),
                        draws,
                        stream_seed(future_seed, "category", category),
                    )
                )
                categories.append(
                    {
                        "category": category,
                        "retry_number": retry,
                        "posterior_file": posterior_file.relative_to(folder).as_posix(),
                    }
                )
                break
        else:
            raise FitFailure(
                f"Independent category {category} failed all frozen budgets.", attempts
            )
    write_json(
        folder / "predictive_checks.json",
        {
            "method": method,
            "origin": origin,
            "per_category": checks,
            "interpretation": "in-sample checks, not out-of-sample calibration",
        },
    )
    return combine_forecasts(forecasts), {
        "fit_status": "accepted",
        "retry_number": max(c["retry_number"] for c in categories),
        "attempts": attempts,
        "per_category": categories,
        "parameter_uncertainty": "separate exact category posteriors",
        "cross_category_dependence": "independent",
        "posterior_layout": "separate_categories",
        "prior_rule": INDEPENDENT_PRIOR_RULE,
    }
