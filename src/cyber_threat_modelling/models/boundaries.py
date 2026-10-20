"""Declared training-only boundary rule shared by ARIMA and future ETS adapters."""

from ..forecasting.count_laws import integer_counts

ZERO_SERIES_RULE = "all_zero_training_point_mass_v1"


def uses_zero_boundary(training_series, method):
    counts = integer_counts(training_series)
    if counts.ndim != 1 or not len(counts):
        raise ValueError("A boundary decision requires a nonempty observed training series.")
    return method in ("arima", "ets") and not counts.any()
