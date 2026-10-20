"""Frozen historical-mean, seasonal-naive and nonseasonal ARIMA count adapters."""

import hashlib
import json
import time
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from ..forecasting.count_laws import CountForecast, integer_counts, round_counts
from .boundaries import ZERO_SERIES_RULE, uses_zero_boundary


def stream_seed(seed, *parts):
    payload = json.dumps([seed, *parts], separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def arima_candidates():
    candidates = []
    for p in range(4):
        for q in range(4):
            if p + q > 4:
                continue
            for d in range(3):
                for trend in ("n", "c") if d == 0 else ("n", "t") if d == 1 else ("n",):
                    candidates.append({"method": "arima", "order": [p, d, q], "trend": trend})
    return candidates


def validate_method(spec):
    if spec in (
        {"method": "historical_mean"},
        {"method": "seasonal_naive"},
        {"method": "independent_nb_ar1"},
        {"method": "pooled_nb"},
    ):
        return
    if spec in arima_candidates():
        return
    raise ValueError("Method settings are outside the implemented frozen baseline definitions.")


def method_id(spec):
    validate_method(spec)
    return (
        "arima_" + "_".join(map(str, spec["order"])) + "_" + spec["trend"]
        if spec["method"] == "arima"
        else spec["method"]
    )


class FitFailure(ValueError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def fit_arima(y, spec):
    model = ARIMA(
        y,
        order=tuple(spec["order"]),
        trend=spec["trend"],
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    attempts, params = [], None
    for number, (optimizer, iterations) in enumerate(
        (("lbfgs", 1000), ("powell", 2000), ("lbfgs", 1000))
    ):
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                fit = model.fit(
                    start_params=params,
                    method="statespace",
                    method_kwargs={"method": optimizer, "maxiter": iterations, "disp": 0},
                )
                valid = bool(
                    fit.mle_retvals.get("converged", False)
                    and np.isfinite(fit.llf)
                    and np.isfinite(fit.params).all()
                )
                if np.isfinite(fit.params).all():
                    params = fit.params
                error = None
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                valid, error = False, type(exc).__name__ + ": " + str(exc)
        attempts.append(
            {
                "retry_number": number,
                "optimizer": optimizer,
                "maxiter": iterations,
                "converged": valid,
                "seconds": time.perf_counter() - started,
                "warnings": [str(w.message) for w in captured],
                "error": error,
            }
        )
        # statsmodels may reset invalid starting AR/MA guesses before optimization.
        # Other warnings require resolution, even if an optimizer reports success.
        unresolved = [
            str(w.message)
            for w in captured
            if not str(w.message).startswith(
                ("Non-stationary starting autoregressive", "Non-invertible starting MA")
            )
        ]
        attempts[-1]["unresolved_warnings"] = unresolved
        valid = valid and not unresolved
        attempts[-1]["accepted_numerics"] = valid
        # A successful Powell retry still requires the frozen L-BFGS refinement.
        if valid and optimizer == "lbfgs":
            return fit, attempts
    raise FitFailure("ARIMA failed the frozen optimizer sequence.", attempts)


def fit_predict(train, category_ids, spec, draws, origin, seed=20261020):
    validate_method(spec)
    y = integer_counts(train)
    if y.ndim != 2 or len(y) < 60 or y.shape[1] != len(category_ids):
        raise ValueError("Baselines need at least 60 complete monthly observations.")
    name = method_id(spec)
    started = time.perf_counter()
    h, k = 12, len(category_ids)
    paths = np.empty((draws, h, k), dtype=np.int64)
    means, scales = np.empty((h, k)), np.empty((h, k))
    records = []
    fitted_series = []
    if spec["method"] == "arima":
        # Complete fitting first: do not simulate a partial panel after a failed fit.
        for j, category in enumerate(category_ids):
            if uses_zero_boundary(y[:, j], "arima"):
                fitted_series.append(None)
                records.append(
                    {
                        "category": int(category),
                        "fit_kind": "declared_boundary_point_mass",
                        "boundary_rule": ZERO_SERIES_RULE,
                        "training_cutoff": origin,
                        "training_months": len(y),
                        "attempts": [],
                    }
                )
                continue
            try:
                fitted, attempts = fit_arima(y[:, j].astype(float), spec)
            except FitFailure as exc:
                raise FitFailure(
                    f"ARIMA failed at category {category}.",
                    records + [{"category": int(category), "attempts": exc.attempts}],
                ) from exc
            fitted_series.append(fitted)
            records.append(
                {
                    "category": int(category),
                    "attempts": attempts,
                    "parameters": fitted.params.tolist(),
                    "log_likelihood": float(fitted.llf),
                }
            )
    for j, category in enumerate(category_ids):
        rng = np.random.default_rng(stream_seed(seed, name, int(category), origin, "forecast"))
        if name == "historical_mean":
            means[:, j] = (y[:, j].sum() + 0.5) / (len(y) + 0.5)
            paths[:, :, j] = rng.poisson(means[:, j], size=(draws, h))
        elif name == "seasonal_naive":
            means[:, j] = y[-12:, j]
            scales[:, j] = max(
                0.5, float(np.sqrt(np.mean((y[12:, j].astype(float) - y[:-12, j]) ** 2)))
            )
            paths[:, :, j] = round_counts(rng.normal(means[:, j], scales[:, j], size=(draws, h)))
        elif spec["method"] == "arima":
            if fitted_series[j] is None:
                means[:, j], scales[:, j], paths[:, :, j] = 0, 0, 0
                continue
            try:
                fitted = fitted_series[j]
                forecast = fitted.get_forecast(h)
                means[:, j] = np.asarray(forecast.predicted_mean)
                variance = np.asarray(forecast.var_pred_mean)
                if not np.isfinite(variance).all() or (variance < 0).any():
                    raise ValueError("Invalid ARIMA predictive variance.")
                scales[:, j] = np.sqrt(variance)
                # ARIMA extends its own trend regressors; passing exog duplicates them.
                simulated = fitted.simulate(h, anchor="end", repetitions=draws, rng=rng)
                paths[:, :, j] = round_counts(np.asarray(simulated).reshape(h, draws).T)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                raise FitFailure(
                    f"ARIMA prediction failed at category {category}: {exc}",
                    records,
                ) from exc
        else:
            raise ValueError("Use the Bayesian adapter for this method.")
    family = "poisson" if name == "historical_mean" else "rounded_normal"
    parameters = {"rate": means} if family == "poisson" else {"location": means, "scale": scales}
    return CountForecast(family, paths, parameters).validate(), {
        "fit_status": "accepted",
        "retry_number": max([0] + [len(r["attempts"]) - 1 for r in records]),
        "fit_and_simulate_seconds": time.perf_counter() - started,
        "per_category": records,
        "boundary_categories": [r["category"] for r in records if "boundary_rule" in r],
        "parameter_uncertainty": "omitted for fitted parameters; declared boundary where recorded",
        "cross_category_dependence": "independent",
        "seed": seed,
    }
