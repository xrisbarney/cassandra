"NUTS sampler and SVI variational inference entry points."
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_median
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import ClippedAdam
import arviz as az

    # Reject effectively collapsed NUTS warmup.
_MIN_USABLE_STEP_SIZE = 1e-6
_MAX_ADAPTATION_RETRIES = 6


def _mu_r_informed_init(site, mu_r_init):
    "Per-site init: mu_r starts near the empirical per-topic log-count level;"
    if site["type"] == "sample" and not site["is_observed"] and site["name"] == "mu_r":
        return mu_r_init
    return init_to_median(site)


def _data_informed_init_strategy(data: dict, R: int):
    "Build an init strategy that starts mu_r near the data's per-topic log-"
    N = np.asarray(data["N"])                       # (K, T)
    e_t = np.asarray(data.get("e_t", np.zeros(N.shape[1])))
    per_topic_level = np.log(N.mean(axis=1) + 1.0) - float(np.mean(e_t))  # (K,)
    mu_r_init = jnp.asarray(np.broadcast_to(per_topic_level, (R, N.shape[0])).copy())
    return partial(_mu_r_informed_init, mu_r_init=mu_r_init)


def _build_mcmc(model, mcmc_cfg: dict, data: dict, R: int) -> MCMC:
    num_warmup = int(mcmc_cfg.get("num_warmup", 500))
    num_samples = int(mcmc_cfg.get("num_samples", 1000))
    num_chains = int(mcmc_cfg.get("num_chains", 1))
    max_tree_depth = int(mcmc_cfg.get("max_tree_depth", 10))
    target_accept_prob = float(mcmc_cfg.get("target_accept_prob", 0.8))

    # The model's discrete latent regime path (z_t) is marginalized
    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        find_heuristic_step_size=True,
        init_strategy=_data_informed_init_strategy(data, R),
    )

    return MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel" if num_chains > 1 else "sequential",
        progress_bar=True,
    )


_STEP_SIZE_FIELD = "adapt_state.step_size"


def _adapted_step_size(mcmc: MCMC) -> float | None:
    "Mean adapted NUTS step size across sampling-phase draws, or None."
    try:
        extra = mcmc.get_extra_fields()
        vals = extra.get(_STEP_SIZE_FIELD)
        if vals is None:
            return None
        return float(np.mean(np.asarray(vals)))
    except Exception:  # noqa: BLE001 — diagnostic best-effort, never fatal
        return None


def run_nuts(model, data: dict, config: dict) -> az.InferenceData:
    "Run NumPyro's NUTS sampler, automatically retrying with a new seed if"
    mcmc_cfg = config.get("mcmc", {})
    num_chains = int(mcmc_cfg.get("num_chains", 1))
    base_seed = int(mcmc_cfg.get("seed", 0))
    R = int(config.get("model", config).get("R", 1))

    best_idata = None
    best_step = -1.0

    for attempt in range(_MAX_ADAPTATION_RETRIES):
        seed = base_seed + attempt
        mcmc = _build_mcmc(model, mcmc_cfg, data, R)
        rng_key = jax.random.PRNGKey(seed)
        if num_chains > 1:
            rng_key = jax.random.split(rng_key, num_chains)

        mcmc.run(rng_key, data=data, config=config, extra_fields=(_STEP_SIZE_FIELD,))
        idata = az.from_numpyro(mcmc)

        step = _adapted_step_size(mcmc)
        if step is None:
            return idata  # sampler exposes no step_size diagnostic; trust it

        if step >= _MIN_USABLE_STEP_SIZE:
            if attempt > 0:
                print(f"      NUTS recovered on retry {attempt + 1} "
                      f"(seed={seed}, adapted step size={step:.2e}).")
            return idata

        print(f"      Warning: NUTS warmup collapsed (adapted step size="
              f"{step:.2e}, attempt {attempt + 1}/{_MAX_ADAPTATION_RETRIES}); "
              f"retrying with a new seed.")
        if step > best_step:
            best_step, best_idata = step, idata

    print(f"      Warning: NUTS adaptation did not recover after "
          f"{_MAX_ADAPTATION_RETRIES} attempts (best adapted step size="
          f"{best_step:.2e}). Returning the best attempt, but treat this "
          f"posterior with caution — consider more warmup or a different seed.")
    return best_idata


def run_svi(model, data: dict, config: dict) -> tuple:
    "Variational inference using NumPyro SVI with AutoLowRankMultivariateNormal guide."
    model_cfg = config.get("model", {})
    svi_cfg = config.get("svi", {})

    rank = int(model_cfg.get("vi_rank", 10))
    num_steps = int(svi_cfg.get("num_steps", 30000))
    learning_rate = float(svi_cfg.get("learning_rate", 1e-3))
    seed = int(svi_cfg.get("seed", 0))

    guide = AutoLowRankMultivariateNormal(model, rank=rank)

    # Learning rate schedule: linear decay from lr to lr/10
    def lr_schedule(step):
        return learning_rate * (1.0 - 0.9 * step / num_steps)

    optimizer = ClippedAdam(step_size=learning_rate, clip_norm=10.0)

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=4))

    rng_key = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng_key, data=data, config=config)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, data=data, config=config)
        losses.append(float(loss))
        if step % 1000 == 0:
            print(f"SVI step {step:5d}  ELBO: {-loss:.4f}")

    params = svi.get_params(svi_state)
    return guide, params, losses
