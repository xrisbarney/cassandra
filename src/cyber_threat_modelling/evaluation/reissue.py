"""Reissue accepted fits at a larger common ensemble budget without refitting."""

import json

import numpy as np

from ..forecasting.count_laws import CountForecast, round_counts
from ..models.count_baselines import stream_seed
from ..provenance import file_hash
from .backtest import artifact_path


def verify_budget_record(root, reference):
    path = artifact_path(root, reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise ValueError("Reporting-budget record changed.")
    decision = json.loads(path.read_text())
    if (
        decision["reporting_draws"] != 6000
        or decision["check_draws"] != 12000
        or decision["parent_freeze_sha256"] != file_hash(root / "docs/STAGE2_FREEZE.json")
    ):
        raise ValueError("Invalid frozen-budget escalation.")
    evidence = artifact_path(root, decision["evidence"]["path"])
    if file_hash(evidence) != decision["evidence"]["sha256"]:
        raise ValueError("Budget escalation evidence changed.")
    checks = json.loads(evidence.read_text())["checks"]
    if not any(not row["passed"] for row in checks):
        raise ValueError("No recorded ensemble failure justifies escalation.")
    return decision


def gaussian_forecast_covariance(result, months=12):
    """Exact joint conditional covariance from a time-invariant Gaussian state space."""
    ssm = result.model.ssm
    matrices = []
    for name in ("design", "transition", "selection", "state_cov", "obs_cov"):
        value = np.asarray(getattr(ssm, name))
        if value.shape[-1] != 1:
            raise ValueError("Joint simulation requires time-invariant covariance matrices.")
        matrices.append(value[..., 0])
    design, transition, selection, state_cov, obs_cov = matrices
    if design.shape[0] != 1:
        raise ValueError("Univariate ARIMA only.")
    variance = result.filter_results.predicted_state_cov[:, :, -1].copy()
    innovation = selection @ state_cov @ selection.T
    covariance = np.empty((months, months))
    for h in range(months):
        covariance[h, h] = (design @ variance @ design.T + obs_cov).item()
        cross = variance.copy()
        for later in range(h + 1, months):
            cross = cross @ transition.T
            covariance[h, later] = covariance[later, h] = (design @ cross @ design.T).item()
        variance = transition @ variance @ transition.T + innovation
    marginal = np.asarray(result.get_forecast(months).var_pred_mean)
    if not np.allclose(np.diag(covariance), marginal, rtol=1e-9, atol=1e-9):
        raise ValueError("Joint covariance disagrees with analytic marginal variances.")
    return covariance


def replay_factory(folder, record, spec, wide, categories):
    """Return a deterministic sampler of the already fitted probability law."""
    name, origin, seed = record["method"], record["origin"], record["seed"]
    saved = CountForecast.load(folder / "forecast.npz")
    fit = json.loads((folder / "fit.json").read_text())
    if name in ("pooled_nb", "independent_nb_glar"):
        import arviz as az

        from ..models.bayesian_controls import posterior_forecast

        idata = az.from_netcdf(folder / f"posterior_attempt_{fit['retry_number']}.nc")
        posterior = idata.posterior.load()
        idata.close()

        def generate(draws):
            return posterior_forecast(
                posterior,
                name,
                len(wide.loc[:origin]),
                draws,
                stream_seed(seed, name, origin, "future"),
                fit.get("last_counts"),
            )

        if not np.array_equal(generate(3000).paths, saved.paths):
            raise ValueError("Source Bayesian forecast fails exact replay.")
        return generate

    factors = {}
    if spec["method"] == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        for j, category in enumerate(categories):
            entry = fit["per_category"][j]
            if entry["category"] != category:
                raise ValueError("Category order changed.")
            if "boundary_rule" in entry:
                continue
            result = ARIMA(
                wide.loc[:origin, category].to_numpy(dtype=float),
                order=tuple(spec["order"]),
                trend=spec["trend"],
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).filter(entry["parameters"])
            prediction = result.get_forecast(12)
            covariance = gaussian_forecast_covariance(result)
            if not np.allclose(prediction.predicted_mean, saved.parameters["location"][:, j]):
                raise ValueError("Saved ARIMA conditional means changed.")
            if not np.allclose(np.diag(covariance), saved.parameters["scale"][:, j] ** 2):
                raise ValueError("Saved ARIMA conditional variances changed.")
            factors[j] = np.linalg.cholesky(covariance)

    def generate(draws):
        paths = np.empty((draws, 12, len(categories)), dtype=np.int64)
        for j, category in enumerate(categories):
            rng = np.random.default_rng(stream_seed(seed, name, int(category), origin, "forecast"))
            if saved.family == "poisson":
                paths[:, :, j] = rng.poisson(saved.parameters["rate"][:, j], size=(draws, 12))
            elif spec["method"] == "arima":
                paths[:, :, j] = (
                    round_counts(
                        saved.parameters["location"][:, j]
                        + rng.normal(size=(draws, 12)) @ factors[j].T
                    )
                    if j in factors
                    else 0
                )
            else:
                paths[:, :, j] = round_counts(
                    rng.normal(
                        saved.parameters["location"][:, j],
                        saved.parameters["scale"][:, j],
                        size=(draws, 12),
                    )
                )
        return CountForecast(saved.family, paths, saved.parameters).validate()

    if spec["method"] != "arima" and not np.array_equal(generate(3000).paths, saved.paths):
        raise ValueError("Source fixed-control forecast fails exact replay.")
    return generate
