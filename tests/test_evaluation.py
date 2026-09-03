"Tests for forecast evaluation utilities."
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.evaluation.scoring import crps_ensemble, mae, rmse
from cassandra_threatcast.evaluation.dm_test import dm_test
from cassandra_threatcast.evaluation.calibration import pit_values, coverage, pit_ks_test


# crps_ensemble

def test_crps_deterministic_forecast():
    "CRPS of a perfect deterministic forecast (all samples = obs) is 0."
    rng = np.random.default_rng(0)
    obs = np.array([1.0, 2.0, 3.0])
    # All 100 samples are exactly equal to the observation
    samples = np.stack([obs] * 100, axis=-1)   # (3, 100)
    crps_vals = crps_ensemble(obs, samples)
    np.testing.assert_allclose(crps_vals, 0.0, atol=1e-10)


def test_crps_ordering():
    "Better (tighter) forecast has lower average CRPS than worse forecast."
    rng = np.random.default_rng(7)
    obs = np.array([5.0, 5.0, 5.0])
    good_samples = rng.normal(5.0, 0.5, (3, 500))
    bad_samples  = rng.normal(5.0, 5.0, (3, 500))
    crps_good = float(np.mean(crps_ensemble(obs, good_samples)))
    crps_bad  = float(np.mean(crps_ensemble(obs, bad_samples)))
    assert crps_good < crps_bad, (
        f"Good CRPS {crps_good:.4f} should be < bad CRPS {crps_bad:.4f}"
    )


def test_crps_analytical():
    "CRPS of N(0, 1) evaluated at y = 0 equals (sqrt(2) - 1)/sqrt(pi) ~= 0.2337."
    rng = np.random.default_rng(42)
    n_samples = 100_000
    obs = np.array([0.0])
    samples = rng.standard_normal((1, n_samples))
    crps_val = crps_ensemble(obs, samples)
    expected = (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi)
    np.testing.assert_allclose(float(crps_val[0]), expected, atol=0.01)


def test_crps_nonnegative():
    "CRPS is always >= 0."
    rng = np.random.default_rng(3)
    obs = rng.standard_normal(20)
    samples = rng.standard_normal((20, 200))
    crps_vals = crps_ensemble(obs, samples)
    assert np.all(crps_vals >= -1e-10), (
        f"CRPS has negative values: min={crps_vals.min():.6f}"
    )


def test_crps_batch_shapes():
    "crps_ensemble handles multi-dimensional obs/samples correctly."
    rng = np.random.default_rng(4)
    obs     = rng.standard_normal((4, 6))       # (K, T)
    samples = rng.standard_normal((4, 6, 100))  # (K, T, n_samples)
    crps_vals = crps_ensemble(obs, samples)
    assert crps_vals.shape == (4, 6)


# dm_test

def test_dm_test_correct_sign():
    "DM stat is negative when model A clearly outperforms model B."
    rng = np.random.default_rng(0)
    T = 100
    obs = rng.standard_normal(T)
    # A: nearly perfect forecasts; B: random noise forecasts
    loss_a = (obs - obs) ** 2 + 0.01
    loss_b = (obs - rng.standard_normal(T)) ** 2
    stat, pval = dm_test(loss_a, loss_b, h=1)
    assert stat < 0.0, f"Expected negative DM stat (A better), got {stat:.3f}"
    assert pval < 0.05, f"Expected significant p-value, got {pval:.4f}"


def test_dm_test_symmetric():
    "Swapping loss_a and loss_b exactly negates the DM statistic."
    rng = np.random.default_rng(1)
    T = 80
    loss_a = rng.exponential(1.0, T)
    loss_b = rng.exponential(2.0, T)
    stat_ab, _ = dm_test(loss_a, loss_b, h=1)
    stat_ba, _ = dm_test(loss_b, loss_a, h=1)
    np.testing.assert_allclose(stat_ab, -stat_ba, atol=1e-10)


def test_dm_test_equal_losses():
    "DM test with identical loss series gives stat near 0 and large p-value."
    rng = np.random.default_rng(2)
    T = 60
    losses = rng.exponential(1.0, T)
    stat, pval = dm_test(losses, losses.copy(), h=1)
    assert abs(stat) < 1e-8, f"Stat should be 0 for equal losses, got {stat}"
    assert pval > 0.99, f"p-value should be near 1.0, got {pval}"


def test_dm_test_raises_mismatched_lengths():
    "dm_test raises ValueError for loss series of different lengths."
    with pytest.raises(ValueError):
        dm_test(np.ones(50), np.ones(40))


def test_dm_test_multi_horizon():
    "DM test with h > 1 runs without error and returns valid stat/pval."
    rng = np.random.default_rng(5)
    T = 120
    loss_a = rng.exponential(1.0, T)
    loss_b = rng.exponential(1.2, T)
    stat, pval = dm_test(loss_a, loss_b, h=6)
    assert np.isfinite(stat), "DM stat should be finite"
    assert 0.0 <= pval <= 1.0, f"p-value out of [0,1]: {pval}"


# pit_values & coverage

def test_pit_coverage_calibrated():
    "90 % interval from a correctly specified model achieves ~90 % coverage."
    rng = np.random.default_rng(42)
    n = 500
    true_mu = rng.standard_normal(n)
    obs     = true_mu + rng.standard_normal(n) * 0.5
    samples = rng.normal(true_mu[:, None], 0.5, (n, 1000))
    lo = np.quantile(samples, 0.05, axis=1)
    hi = np.quantile(samples, 0.95, axis=1)
    cov = coverage(obs, lo, hi)
    assert 0.85 < cov < 0.97, f"Coverage {cov:.3f} should be near 0.90"


def test_pit_uniform_correct_model():
    "PIT values from a correctly specified model pass the KS uniformity test."
    rng = np.random.default_rng(99)
    n = 1000
    obs     = rng.standard_normal(n)
    samples = rng.standard_normal((n, 500))
    pit_vals = pit_values(obs, samples)
    stat, pval = pit_ks_test(pit_vals)
    # Should fail to reject uniformity at 1 % level for a correct model
    assert pval > 0.01, (
        f"KS test rejected uniformity (stat={stat:.3f}, p={pval:.4f}) "
        "for a correctly specified model"
    )


def test_pit_biased_model():
    "PIT values from a biased model fail the KS uniformity test."
    rng = np.random.default_rng(10)
    n = 500
    obs     = rng.standard_normal(n)
    # Samples centred at +3 â€” deliberately wrong
    samples = rng.normal(3.0, 1.0, (n, 500))
    pit_vals = pit_values(obs, samples)
    stat, pval = pit_ks_test(pit_vals)
    assert pval < 0.05, (
        f"KS test should reject uniformity for a biased model "
        f"(stat={stat:.3f}, p={pval:.4f})"
    )


# mae and rmse

def test_mae_rmse_correct():
    "mae and rmse give analytically correct values for a simple case."
    obs  = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    assert mae(obs, pred)  == pytest.approx(0.5, abs=1e-10)
    assert rmse(obs, pred) == pytest.approx(0.5, abs=1e-10)


def test_mae_rmse_perfect_forecast():
    "mae and rmse are 0 for a perfect point forecast."
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(obs, obs)  == pytest.approx(0.0, abs=1e-10)
    assert rmse(obs, obs) == pytest.approx(0.0, abs=1e-10)


def test_rmse_ge_mae():
    "RMSE >= MAE always holds (by Jensen's inequality)."
    rng = np.random.default_rng(11)
    obs  = rng.poisson(5, 100).astype(float)
    pred = rng.poisson(5, 100).astype(float)
    assert rmse(obs, pred) >= mae(obs, pred) - 1e-10
