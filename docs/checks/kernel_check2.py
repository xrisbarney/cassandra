"""Check the stated kernel approximation; this is not a fitted threat model.

Run from the project root: python docs/checks/kernel_check2.py
The widths and lags below form a finite numerical check, not a uniform bound.
"""

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def weights(width, lags, centre=0):
    raw = [sigmoid((width - abs(d - centre)) / 2) for d in lags]
    norm = math.sqrt(math.fsum(value * value for value in raw))
    return [value / norm for value in raw]


def covariance(values, lag):
    return math.fsum(a * b for a, b in zip(values, values[lag:]))


print("Finite checks: widths 6, 12, 12.5, 18; comparison lags 0..60.")
print("Two-sided reference support: -300..300; causal support: 0..60.")
for width in (6.0, 12.0, 12.5, 18.0):
    reference = weights(width, range(-300, 301))
    truncated = weights(width, range(-30, 31))
    causal = weights(width, range(61), centre=30)
    # The integer shift is exact for equal truncation and normalization.
    assert causal == truncated
    assert math.isclose(covariance(causal, 0), 1.0, abs_tol=1e-14)
    assert covariance(causal, 61) == 0.0
    lag12_error = abs(covariance(reference, 12) - covariance(causal, 12))
    max_error, lag = max(
        (abs(covariance(reference, d) - covariance(causal, d)), d)
        for d in range(61)
    )
    assert lag12_error < 1e-4  # Only the four widths tested here.
    print(
        f"b={width:g}: lag12 reference={covariance(reference, 12):.9f} "
        f"causal={covariance(causal, 12):.9f} error={lag12_error:.9g}; "
        f"max_error_0_to_60={max_error:.9g} at lag={lag}"
    )
    if width == 18:
        assert max_error > 1e-4  # Refutes the earlier all-lag claim.

print(f"b=18 retained edge weight: {sigmoid((18 - 30) / 2):.12f}")
print(f"b=18 largest discarded weight: {sigmoid((18 - 31) / 2):.12f}")
print("All finite-check assertions passed; no likelihood-error bound established.")
