import numpy as np
import pytest

from cyber_threat_modelling.models.independent_controls import (
    combine_forecasts,
    forecast_from_files,
)


def test_rebinding_observations_and_anchor_matches_a_fresh_density():
    import pymc as pm

    from cyber_threat_modelling.models.independent_geometry import build_conditioned_model

    first = np.arange(8)[:, None]
    second = np.zeros((8, 1), dtype=int)
    model = build_conditioned_model(first, first[:4], mutable=True, stationary_scale_sd=0.75)
    logp, grad = model.compile_logp(mode="NUMBA"), model.compile_dlogp(mode="NUMBA")
    pm.set_data({"training_counts": second.T, "prior_mean": np.log([0.5])}, model=model)
    point = model.initial_point()
    fresh = build_conditioned_model(second, second[:4], stationary_scale_sd=0.75)
    assert np.allclose(logp(point), fresh.compile_logp()(point), atol=1e-10)
    assert np.allclose(grad(point), fresh.compile_dlogp()(point), atol=1e-9)
    assert np.array_equal(point["initial_eta"], np.log([0.5]))


def test_amended_stationary_scale_prior_and_initial_state():
    import pymc as pm

    from cyber_threat_modelling.models.independent_geometry import build_conditioned_model

    y = np.full((8, 1), 10)
    model = build_conditioned_model(y, y[:4], stationary_scale_sd=0.75)
    with model:
        draws = pm.sample_prior_predictive(
            draws=12000,
            random_seed=861,
            var_names=["m", "phi", "tau", "stationary_scale", "unit_state"],
        ).prior
    s, phi, tau = draws.stationary_scale.values, draws.phi.values, draws.tau.values
    u0 = (draws.tau * draws.unit_state.isel(month=0)).values
    assert np.allclose(tau**2, s**2 * (1 - phi**2))
    assert abs(np.mean(s**2) - 0.75**2) < 0.03
    assert abs(np.mean((u0 / s) ** 2) - 1) < 0.05
    assert abs(np.var(draws.m.values) - 4) < 0.2
    assert abs(np.corrcoef(draws.m.values.ravel(), u0.ravel())[0, 1]) < 0.04


def test_independent_prior_amendment_is_verified_and_tampering_rejected(tmp_path):
    import json
    from pathlib import Path

    from cyber_threat_modelling.provenance import verify_independent_control_amendment

    root = Path(__file__).resolve().parents[1]
    record = verify_independent_control_amendment(root, required=True)
    assert record["record"]["stationary_scale_sd"] == 0.75
    with pytest.raises(ValueError, match="required"):
        verify_independent_control_amendment(tmp_path, required=True)
    target = tmp_path / record["path"]
    target.parent.mkdir(parents=True)
    for entry in record["record"]["files"]:
        p = tmp_path / entry["path"]
        p.write_bytes((root / entry["path"]).read_bytes())
    (tmp_path / "docs/STAGE2_FREEZE.json").write_bytes(
        (root / "docs/STAGE2_FREEZE.json").read_bytes()
    )
    target.write_text(json.dumps(dict(record["record"], stationary_scale_sd=0.9)))
    with pytest.raises(ValueError, match="Invalid independent"):
        verify_independent_control_amendment(tmp_path, required=True)


def test_separate_posteriors_allow_different_draw_counts_without_pairing_categories(tmp_path):
    import arviz as az

    entries = []
    for category, n in ((1, 1000), (2, 2000)):
        shape = (4, n, 1)
        means = np.broadcast_to(np.tile([0.0, 3.0], n // 2)[None, :, None], shape)
        posterior = az.from_dict(
            posterior={
                "m": means,
                "psi": np.full(shape, 3.0),
                "tau": np.zeros(shape),
                "phi": np.zeros(shape),
                "terminal_u": np.zeros(shape),
            },
            dims={key: ["category"] for key in ("m", "psi", "tau", "phi", "terminal_u")},
            coords={"category": [0]},
        )
        path = tmp_path / f"category_{category}.nc"
        posterior.to_netcdf(path)
        entries.append({"category": category, "posterior_file": path.name})
    fit = {"per_category": entries}
    a = forecast_from_files(tmp_path, fit, 108, 3000, 6)
    b = forecast_from_files(tmp_path, fit, 108, 3000, 6)
    assert a.paths.shape == (3000, 12, 2)
    assert np.array_equal(a.paths, b.paths)
    levels = a.parameters["mean"][:, 0, :]
    assert abs(np.corrcoef(levels.T)[0, 1]) < 0.06
    with pytest.raises(ValueError, match="reordered"):
        forecast_from_files(tmp_path, {"per_category": entries[::-1]}, 108, 3000, 6)
    entries[0]["posterior_file"] = "../outside.nc"
    with pytest.raises(ValueError, match="escapes"):
        forecast_from_files(tmp_path, fit, 108, 3000, 6)
    with pytest.raises(ValueError, match="per category"):
        combine_forecasts([])
