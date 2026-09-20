"Live smoke test for the BEA Input-Output API integration."

import os

import numpy as np
import pytest

from cassandra_threatcast.data import bea_io

pytestmark = pytest.mark.skipif(
    not os.environ.get("BEA_API_KEY"),
    reason="BEA_API_KEY not set; skipping live BEA API smoke test.",
)


@pytest.fixture(scope="module")
def A():
    return bea_io.get_technical_coefficients(year=2022, n_sectors=11, cache_dir="data/cache")


def test_matrix_shape_and_nonzero(A):
    "A must be 11x11 and substantially non-zero (not the identity fallback)."
    assert A.shape == (11, 11)
    assert (A > 0).sum() >= 50, "Too few non-zero cells — code→sector map likely broken."


def test_spectral_radius_below_one(A):
    "A productive economy has spectral radius < 1 so the Leontief inverse exists."
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    assert 0.0 < rho < 1.0, f"Spectral radius {rho} outside (0, 1)."


def test_leontief_inverse_amplifies(A):
    "L = (I-A)^-1 must dominate the identity (diagonal >= 1)."
    L = bea_io.leontief_inverse(A)
    assert L.shape == (11, 11)
    assert np.all(np.diag(L) >= 1.0 - 1e-9)


def test_gross_output_realistic():
    "Total US gross output for 2022 is ~$46 trillion (sanity band $30-60T)."
    x = bea_io.get_sector_output(year=2022, n_sectors=11, cache_dir="data/cache")
    total_trillions = x.sum() / 1e6
    assert 30.0 < total_trillions < 60.0, f"Total gross output ${total_trillions:.1f}T out of range."
