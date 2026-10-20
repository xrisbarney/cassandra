"""Issued count laws: saved paths and the same laws' analytic marginal mass/CDF."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import nbinom, norm, poisson


def integer_counts(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Counts must be finite, nonnegative integers.")
    if (values >= 2**63 - 1).any():
        raise ValueError("Count exceeds supported integer range; no tail clipping is permitted.")
    return values.astype(np.int64)


def round_counts(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite forecast; tail repair is forbidden.")
    return integer_counts(np.maximum(0, np.floor(values + 0.5)))


def logdiffexp(a, b):
    """log(exp(a)-exp(b)), where a>=b; equality retains genuine zero mass."""
    with np.errstate(divide="ignore", invalid="ignore", over="raise"):
        return a + np.log(-np.expm1(b - a))


@dataclass
class CountForecast:
    family: str
    paths: np.ndarray  # draw, horizon, panel category
    parameters: dict

    def validate(self):
        self.paths = integer_counts(self.paths)
        if self.paths.ndim != 3 or min(self.paths.shape) < 1:
            raise ValueError("Forecast paths must have draw/horizon/category axes.")
        shape = self.paths.shape[1:]
        if self.family == "poisson":
            required = {"rate": shape}
        elif self.family == "rounded_normal":
            required = {"location": shape, "scale": shape}
        elif self.family == "nb2_mixture":
            required = {"mean": self.paths.shape, "concentration": self.paths.shape}
        else:
            raise ValueError("Unsupported issued count distribution.")
        if set(required) != set(self.parameters):
            raise ValueError("Count law parameters do not match its family.")
        for key, expected in required.items():
            arr = np.asarray(self.parameters[key], dtype=float)
            if arr.shape != expected or not np.isfinite(arr).all():
                raise ValueError("Invalid distribution parameter shape or values.")
            if key != "location" and (arr < 0).any():
                raise ValueError("Count law scale/rate must be nonnegative.")
            if key in ("mean", "concentration") and (arr <= 0).any():
                raise ValueError("NB mixture parameters must be positive.")
            self.parameters[key] = arr
        return self

    def logpmf(self, y, h, k):
        y = int(integer_counts([y])[0])
        p = self.parameters
        if self.family == "poisson":
            return float(poisson.logpmf(y, p["rate"][h, k]))
        if self.family == "nb2_mixture":
            mu, a = p["mean"][:, h, k], p["concentration"][:, h, k]
            # Log-domain NB2 mass avoids rounding a/(a+mu) to one for tiny means.
            total = np.logaddexp(np.log(a), np.log(mu))
            components = (
                gammaln(y + a)
                - gammaln(a)
                - gammaln(y + 1)
                + a * (np.log(a) - total)
                + y * (np.log(mu) - total)
            )
            return float(logsumexp(components) - np.log(len(mu)))
        loc, scale = p["location"][h, k], p["scale"][h, k]
        if scale == 0:
            return 0.0 if y == round_counts([loc])[0] else -np.inf
        hi = (y + 0.5 - loc) / scale
        if y == 0:
            return float(log_ndtr(hi))
        lo = (y - 0.5 - loc) / scale
        # Choose the tail representation that avoids subtraction near one.
        return float(
            logdiffexp(log_ndtr(-lo), log_ndtr(-hi))
            if lo > 0
            else logdiffexp(log_ndtr(hi), log_ndtr(lo))
        )

    def cdf(self, y, h, k):
        if y < 0:
            return 0.0
        y = np.floor(y)
        p = self.parameters
        if self.family == "poisson":
            return float(poisson.cdf(y, p["rate"][h, k]))
        if self.family == "nb2_mixture":
            mu, a = p["mean"][:, h, k], p["concentration"][:, h, k]
            return float(np.mean(nbinom.cdf(y, a, a / (a + mu))))
        loc, scale = p["location"][h, k], p["scale"][h, k]
        return (
            float(y >= round_counts([loc])[0])
            if scale == 0
            else float(norm.cdf((y + 0.5 - loc) / scale))
        )

    def quantile(self, q, h, k):
        if not 0 < q < 1:
            raise ValueError("Quantile probability must be interior.")
        if self.cdf(0, h, k) >= q:
            return 0
        low, high = 0, 1
        while self.cdf(high, h, k) < q:
            high *= 2
            if high >= 2**62:
                raise ValueError("Count quantile exceeds supported range.")
        while high - low > 1:
            middle = (low + high) // 2
            if self.cdf(middle, h, k) >= q:
                high = middle
            else:
                low = middle
        return high

    def save(self, path):
        self.validate()
        with open(path, "xb") as out:
            np.savez_compressed(
                out, family=np.array(self.family), paths=self.paths, **self.parameters
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            return cls(
                str(state["family"]),
                state["paths"],
                {k: state[k] for k in state.files if k not in ("family", "paths")},
            ).validate()
