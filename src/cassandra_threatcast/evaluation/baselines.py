"""Baseline forecasting models for comparison against the full Bayesian model."""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class RandomForestBaseline:
    """
    Per-series Random Forest with lag-embedded features.

    Quantiles are derived from the distribution of individual tree predictions
    (i.e., the ensemble distribution across estimators), following
    Meinshausen (2006) quantile regression forests.
    """

    def __init__(
        self,
        n_lags: int = 12,
        n_estimators: int = 200,
        quantiles: list = [0.1, 0.5, 0.9],
    ):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.quantiles = quantiles
        self.models: dict = {}  # series_idx -> fitted RF

    def _make_features(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create lag features X and targets y from a univariate series."""
        T = len(series)
        X, y = [], []
        for t in range(self.n_lags, T):
            X.append(series[t - self.n_lags : t])
            y.append(series[t])
        return np.array(X), np.array(y)

    def fit(self, panel: np.ndarray) -> "RandomForestBaseline":
        """
        Fit one RF per series.

        panel : (n_series, T) array of observed counts
        """
        n_series = panel.shape[0]
        for i in range(n_series):
            X, y = self._make_features(panel[i])
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=42, n_jobs=-1
            )
            rf.fit(X, y)
            self.models[i] = rf
        return self

    def predict_quantiles(
        self,
        panel: np.ndarray,
        horizon: int,
        quantiles: list | None = None,
    ) -> np.ndarray:
        """
        Generate quantile forecasts by rolling the RF forward.

        Uses the distribution of individual tree predictions as the
        predictive distribution; point forecast for the next step
        is the median over trees.

        Returns : (n_series, horizon, n_quantiles)
        """
        if quantiles is None:
            quantiles = self.quantiles
        n_series = panel.shape[0]
        results = np.zeros((n_series, horizon, len(quantiles)))

        for i, rf in self.models.items():
            # Seed the rolling context with the last n_lags observations
            context = list(panel[i, -self.n_lags :])
            for h in range(horizon):
                feat = np.array(context[-self.n_lags :]).reshape(1, -1)
                # One prediction per tree
                tree_preds = np.array([tree.predict(feat)[0] for tree in rf.estimators_])
                for qi, q in enumerate(quantiles):
                    results[i, h, qi] = np.quantile(tree_preds, q)
                # Advance context using the median tree prediction
                context.append(float(np.median(tree_preds)))

        return results


class ArimaBaseline:
    """
    Per-series ARIMA via pmdarima auto_arima.

    Quantile intervals are obtained by simulating from a Normal distribution
    whose standard deviation is derived from the 90 % prediction interval
    returned by pmdarima.
    """

    def __init__(self, seasonal: bool = True, m: int = 12):
        self.seasonal = seasonal
        self.m = m
        self.models: dict = {}

    def fit(self, panel: np.ndarray) -> "ArimaBaseline":
        """panel : (n_series, T)"""
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("pmdarima is required: pip install pmdarima")

        n_series = panel.shape[0]
        for i in range(n_series):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = pm.auto_arima(
                    panel[i],
                    seasonal=self.seasonal,
                    m=self.m,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                )
            self.models[i] = model
        return self

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list = [0.1, 0.5, 0.9],
        n_bootstrap: int = 500,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Returns (n_series, horizon, n_quantiles).

        Quantiles are obtained by bootstrap sampling from a Normal(fc[h], sigma[h])
        where sigma[h] is inferred from the 90 % prediction interval width.
        """
        n_series = len(self.models)
        results = np.zeros((n_series, horizon, len(quantiles)))
        rng = np.random.default_rng(seed)

        for i, model in self.models.items():
            fc, conf = model.predict(
                n_periods=horizon, return_conf_int=True, alpha=0.1
            )
            for h in range(horizon):
                # 90% PI: conf[h, 0] = lower (5th pct), conf[h, 1] = upper (95th pct)
                sigma = max((conf[h, 1] - conf[h, 0]) / (2.0 * 1.645), 1e-6)
                samples = rng.normal(fc[h], sigma, n_bootstrap)
                for qi, q in enumerate(quantiles):
                    results[i, h, qi] = np.quantile(samples, q)

        return results


class EtsBaseline:
    """
    Per-series ETS (Holt-Winters) via statsmodels.

    Falls back to SimpleExpSmoothing when the series is too short for
    full seasonal decomposition.
    """

    def __init__(
        self,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 12,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.models: dict = {}

    def fit(self, panel: np.ndarray) -> "EtsBaseline":
        """panel : (n_series, T)"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW

        for i in range(panel.shape[0]):
            try:
                use_seasonal = (
                    self.seasonal
                    if len(panel[i]) > 2 * self.seasonal_periods
                    else None
                )
                model = HW(
                    panel[i],
                    trend=self.trend,
                    seasonal=use_seasonal,
                    seasonal_periods=self.seasonal_periods,
                ).fit(disp=False)
            except Exception:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                model = SimpleExpSmoothing(panel[i]).fit()
            self.models[i] = model
        return self

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list = [0.1, 0.5, 0.9],
        n_bootstrap: int = 500,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Returns (n_series, horizon, n_quantiles).

        Uncertainty grows with horizon via sqrt(h + 1) scaling of the
        residual standard deviation.
        """
        n_series = len(self.models)
        results = np.zeros((n_series, horizon, len(quantiles)))
        rng = np.random.default_rng(seed)

        for i, model in self.models.items():
            fc = model.forecast(horizon)
            if hasattr(model, "resid") and model.resid is not None:
                resid_std = max(float(np.std(model.resid)), 1e-6)
            else:
                resid_std = 1.0

            for h in range(horizon):
                scale = resid_std * np.sqrt(h + 1)
                samples = rng.normal(fc[h], scale, n_bootstrap)
                samples = np.maximum(samples, 0.0)
                for qi, q in enumerate(quantiles):
                    results[i, h, qi] = np.quantile(samples, q)

        return results


class NaiveBaseline:
    """
    Seasonal random walk: forecast = last observed same-season value
    plus additive Gaussian noise whose variance grows with the number
    of full seasonal cycles elapsed.
    """

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self.panel: np.ndarray | None = None

    def fit(self, panel: np.ndarray) -> "NaiveBaseline":
        """panel : (n_series, T)"""
        self.panel = panel.copy()
        return self

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list = [0.1, 0.5, 0.9],
        n_bootstrap: int = 500,
        seed: int = 0,
    ) -> np.ndarray:
        """Returns (n_series, horizon, n_quantiles)."""
        n_series = self.panel.shape[0]
        results = np.zeros((n_series, horizon, len(quantiles)))
        rng = np.random.default_rng(seed)

        for i in range(n_series):
            series = self.panel[i]
            T = len(series)

            # Estimate residual noise from seasonal differences
            if T > self.seasonal_period:
                seas_diff = series[self.seasonal_period :] - series[: T - self.seasonal_period]
                resid_std = max(float(np.std(np.diff(seas_diff))), 1e-6) if len(seas_diff) > 1 else 1.0
            else:
                resid_std = float(np.std(series)) if T > 1 else 1.0

            for h in range(horizon):
                # Index of the observation from the same season
                offset = (h % self.seasonal_period) + 1
                idx = -self.seasonal_period + (h % self.seasonal_period)
                if idx >= 0:
                    idx = -(self.seasonal_period - (h % self.seasonal_period))
                point = series[idx] if abs(idx) <= T else float(np.mean(series))
                # Noise grows with the number of full seasonal cycles
                scale = resid_std * np.sqrt(h // self.seasonal_period + 1)
                samples = rng.normal(point, scale, n_bootstrap)
                samples = np.maximum(samples, 0.0)
                for qi, q in enumerate(quantiles):
                    results[i, h, qi] = np.quantile(samples, q)

        return results


class BstsUnivariate:
    """
    Per-series BSTS: local linear trend estimated in NumPyro (NUTS).

    This is a univariate baseline without topic factors or regime switching.
    Posterior predictive samples are propagated forward using the last
    level and trend draws.
    """

    def __init__(self, num_warmup: int = 500, num_samples: int = 1000):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.posterior: dict = {}
        self._T_fit: int = 0

    def _model(self, T: int, obs=None):
        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp
        from numpyro.contrib.control_flow import scan

        level_std = numpyro.sample("level_std", dist.HalfNormal(1.0))
        trend_std = numpyro.sample("trend_std", dist.HalfNormal(0.1))
        obs_std = numpyro.sample("obs_std", dist.HalfNormal(1.0))
        level_0 = numpyro.sample("level_0", dist.Normal(0.0, 10.0))
        trend_0 = numpyro.sample("trend_0", dist.Normal(0.0, 1.0))

        def scan_fn(carry, t):
            level, trend = carry
            new_level = numpyro.sample(
                "level", dist.Normal(level + trend, level_std)
            )
            new_trend = numpyro.sample(
                "trend", dist.Normal(trend, trend_std)
            )
            return (new_level, new_trend), new_level

        _, levels = scan(scan_fn, (level_0, trend_0), jnp.arange(T))
        numpyro.sample("obs", dist.Normal(levels, obs_std), obs=obs)

    def fit(self, series: np.ndarray, seed: int = 0) -> "BstsUnivariate":
        """Fit via NUTS; stores posterior samples."""
        import numpyro
        from numpyro.infer import MCMC, NUTS
        import jax
        import jax.numpy as jnp

        T = len(series)
        obs_jnp = jnp.array(series, dtype=float)

        nuts = NUTS(self._model)
        mcmc = MCMC(
            nuts,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed), T=T, obs=obs_jnp)
        self.posterior = {k: np.array(v) for k, v in mcmc.get_samples().items()}
        self._T_fit = T
        return self

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list = [0.1, 0.5, 0.9],
    ) -> np.ndarray:
        """
        Propagate posterior level and trend forward for `horizon` steps.

        Returns : (horizon, n_quantiles)
        """
        level_std = self.posterior["level_std"]   # (n_samples,)
        trend_std = self.posterior["trend_std"]
        obs_std = self.posterior["obs_std"]

        # levels shape: (n_samples, T); take last time step
        levels = self.posterior["level"]
        trends = self.posterior["trend"]
        last_level = levels[:, -1].copy()
        last_trend = trends[:, -1].copy()

        n_samples = len(last_level)
        preds = np.zeros((n_samples, horizon))

        lv = last_level.copy()
        tr = last_trend.copy()

        rng = np.random.default_rng(0)
        for h in range(horizon):
            lv = lv + tr + rng.standard_normal(n_samples) * level_std
            tr = tr + rng.standard_normal(n_samples) * trend_std
            preds[:, h] = lv + rng.standard_normal(n_samples) * obs_std

        results = np.zeros((horizon, len(quantiles)))
        for h in range(horizon):
            for qi, q in enumerate(quantiles):
                results[h, qi] = np.quantile(preds[:, h], q)
        return results


def run_all_baselines(
    panel: np.ndarray,
    horizons: list[int],
    quantiles: list[float] = [0.1, 0.5, 0.9],
    test_T: int = 12,
    seed: int = 0,
) -> dict:
    """
    Fit all baselines on panel[:, :-test_T] and predict for test_T steps.

    Parameters
    ----------
    panel     : (n_series, T) observed counts
    horizons  : list of horizon lengths (informational; stored in results)
    quantiles : quantile levels to forecast
    test_T    : number of held-out time steps
    seed      : RNG seed for baselines' bootstrap quantile sampling (ETS,
                Naive, ARIMA -- RF's predictive spread comes from its trees,
                not bootstrap noise, so it ignores this). Each baseline gets
                a distinct seed (seed, seed+1, seed+2) so their bootstrap
                draws aren't identical to each other.

    Returns
    -------
    dict mapping baseline name ->
        {"predictions": (n_series, test_T, n_quantiles), "name": str}
    """
    train = panel[:, :-test_T]
    results = {}

    # --- Random Forest ---
    try:
        rf = RandomForestBaseline(quantiles=quantiles).fit(train)
        preds = rf.predict_quantiles(train, test_T, quantiles)
        results["RF"] = {"predictions": preds, "name": "RF"}
    except Exception as e:
        print(f"Warning: RF baseline failed: {e}")

    # --- ETS ---
    try:
        ets = EtsBaseline().fit(train)
        preds = ets.predict_quantiles(test_T, quantiles, seed=seed)
        results["ETS"] = {"predictions": preds, "name": "ETS"}
    except Exception as e:
        print(f"Warning: ETS baseline failed: {e}")

    # --- Naive ---
    try:
        naive = NaiveBaseline().fit(train)
        preds = naive.predict_quantiles(test_T, quantiles, seed=seed + 1)
        results["Naive"] = {"predictions": preds, "name": "Naive"}
    except Exception as e:
        print(f"Warning: Naive baseline failed: {e}")

    # --- ARIMA ---
    try:
        arima = ArimaBaseline().fit(train)
        preds = arima.predict_quantiles(test_T, quantiles, seed=seed + 2)
        results["ARIMA"] = {"predictions": preds, "name": "ARIMA"}
    except Exception as e:
        print(f"Warning: ARIMA baseline failed: {e}")

    return results
