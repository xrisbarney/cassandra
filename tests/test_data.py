"""Tests for data ingestion and feature utilities."""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cassandra_threatcast.data.bea_io import leontief_inverse, get_default_sector_labels
from cassandra_threatcast.data.nvd import aggregate_monthly
from cassandra_threatcast.features.effort import estimate_effort


# ---------------------------------------------------------------------------
# leontief_inverse
# ---------------------------------------------------------------------------

def test_leontief_identity():
    """Leontief inverse of a zero technical-coefficient matrix is the identity."""
    A = np.zeros((5, 5))
    L = leontief_inverse(A)
    np.testing.assert_allclose(L, np.eye(5), atol=1e-10)


def test_leontief_raises_on_unstable():
    """leontief_inverse raises ValueError when the spectral radius of A >= 1."""
    A = 2.0 * np.eye(5)   # spectral radius = 2
    with pytest.raises(ValueError, match="(?i)spectral radius"):
        leontief_inverse(A)


def test_leontief_simple_chain():
    """Verify Leontief inverse for a 2-sector chain: A = [[0, 0.5], [0, 0]]."""
    A = np.array([[0.0, 0.5],
                  [0.0, 0.0]])
    L = leontief_inverse(A)
    expected = np.array([[1.0, 0.5],
                         [0.0, 1.0]])
    np.testing.assert_allclose(L, expected, atol=1e-10)


def test_leontief_positive_entries():
    """Leontief inverse has all entries >= 1 on the diagonal and >= 0 off-diagonal."""
    rng = np.random.default_rng(7)
    # Stable A: each column sums to < 1
    A = rng.uniform(0, 0.15, (4, 4))
    L = leontief_inverse(A)
    # Diagonal entries must be >= 1
    assert np.all(np.diag(L) >= 1.0 - 1e-10)
    # All entries must be non-negative
    assert np.all(L >= -1e-10)


# ---------------------------------------------------------------------------
# aggregate_monthly
# ---------------------------------------------------------------------------

def test_aggregate_monthly_shape():
    """aggregate_monthly returns arrays of shape (K, T)."""
    K, T = 4, 24
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=T * 10, freq="3D")
    df = pd.DataFrame(
        {
            "cve_id": [f"CVE-2022-{i:04d}" for i in range(len(dates))],
            "published_date": dates,
            "cvss_base_score": rng.uniform(1.0, 10.0, len(dates)),
        }
    ).set_index("cve_id")
    topic_assignments = rng.integers(0, K, len(df))
    N_kt, B_kt = aggregate_monthly(df, topic_assignments)
    assert N_kt.shape[0] == K
    assert N_kt.shape[1] >= 1
    assert B_kt.shape == N_kt.shape


def test_aggregate_monthly_nonneg():
    """aggregate_monthly counts and severity scores are non-negative."""
    K = 3
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=60, freq="5D")
    df = pd.DataFrame(
        {
            "cve_id": [f"CVE-{i}" for i in range(len(dates))],
            "published_date": dates,
            "cvss_base_score": rng.uniform(1.0, 10.0, len(dates)),
        }
    ).set_index("cve_id")
    topic_assignments = rng.integers(0, K, len(df))
    N_kt, B_kt = aggregate_monthly(df, topic_assignments)
    assert np.all(N_kt >= 0)
    # Empty (topic, month) cells carry NaN mean severity by design; where a
    # severity is defined it must be a non-negative CVSS score.
    assert np.all(B_kt[~np.isnan(B_kt)] >= 0)


def test_aggregate_monthly_total_counts():
    """Total counts across topics should equal total number of CVEs."""
    K = 5
    rng = np.random.default_rng(2)
    dates = pd.date_range("2021-06-01", periods=100, freq="3D")
    df = pd.DataFrame(
        {
            "cve_id": [f"CVE-{i}" for i in range(len(dates))],
            "published_date": dates,
            "cvss_base_score": rng.uniform(1.0, 10.0, len(dates)),
        }
    ).set_index("cve_id")
    topic_assignments = rng.integers(0, K, len(df))
    N_kt, _ = aggregate_monthly(df, topic_assignments)
    assert int(N_kt.sum()) == len(df)


# ---------------------------------------------------------------------------
# estimate_effort
# ---------------------------------------------------------------------------

def test_estimate_effort_length():
    """estimate_effort returns an array of the same length as input."""
    T = 60
    rng = np.random.default_rng(42)
    N_raw = rng.poisson(100, T).astype(float)
    e_t = estimate_effort(N_raw, method="hp_filter")
    assert len(e_t) == T


def test_estimate_effort_normalized():
    """HP-filter effort has approximately zero mean and reasonable variance."""
    T = 120
    rng = np.random.default_rng(0)
    N_raw = rng.poisson(200, T).astype(float) + np.linspace(0.0, 100.0, T)
    e_t = estimate_effort(N_raw, method="hp_filter")
    assert abs(float(np.mean(e_t))) < 0.5
    assert 0.5 < float(np.std(e_t)) < 2.0


def test_estimate_effort_finite():
    """estimate_effort output is finite for all reasonable inputs."""
    T = 48
    rng = np.random.default_rng(3)
    for method in ["hp_filter", "moving_avg", "log_diff"]:
        N_raw = rng.poisson(50, T).astype(float) + 1.0  # avoid zeros for log_diff
        e_t = estimate_effort(N_raw, method=method)
        assert np.all(np.isfinite(e_t)), f"Non-finite values for method={method}"


# ---------------------------------------------------------------------------
# get_default_sector_labels
# ---------------------------------------------------------------------------

def test_sector_labels_count():
    """get_default_sector_labels returns exactly 11 sector labels."""
    labels = get_default_sector_labels()
    assert len(labels) == 11


def test_sector_labels_are_strings():
    """All sector labels are non-empty strings."""
    labels = get_default_sector_labels()
    for lbl in labels:
        assert isinstance(lbl, str) and len(lbl) > 0
