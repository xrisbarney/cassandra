"""
nuts.py
=======
NUTS sampler and SVI variational inference entry points.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import ClippedAdam
import arviz as az


def run_nuts(model, data: dict, config: dict) -> az.InferenceData:
    """
    Run NumPyro's NUTS sampler.

    Parameters
    ----------
    model  : NumPyro model function (signature model(data, config) → None).
    data   : Data dict forwarded to the model.
    config : Config dict; reads config["mcmc"] for:
               num_warmup  (default 500)
               num_samples (default 1000)
               num_chains  (default 1)
               max_tree_depth (default 10)
               target_accept_prob (default 0.8)

    Returns
    -------
    arviz.InferenceData object with posterior and sample stats.
    """
    mcmc_cfg = config.get("mcmc", {})
    num_warmup = int(mcmc_cfg.get("num_warmup", 500))
    num_samples = int(mcmc_cfg.get("num_samples", 1000))
    num_chains = int(mcmc_cfg.get("num_chains", 1))
    max_tree_depth = int(mcmc_cfg.get("max_tree_depth", 10))
    target_accept_prob = float(mcmc_cfg.get("target_accept_prob", 0.8))

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        find_heuristic_step_size=True,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel" if num_chains > 1 else "sequential",
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(mcmc_cfg.get("seed", 0))

    if num_chains > 1:
        # Split key for parallel chains
        rng_key = jax.random.split(rng_key, num_chains)

    mcmc.run(rng_key, data=data, config=config)

    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(mcmc)
    return idata


def run_svi(model, data: dict, config: dict) -> tuple:
    """
    Variational inference using NumPyro SVI with AutoLowRankMultivariateNormal guide.

    Parameters
    ----------
    model  : NumPyro model function.
    data   : Data dict forwarded to the model.
    config : Config dict; reads config["model"]["vi_rank"] (default 10) and
             config["svi"] for num_steps, learning_rate, seed.

    Returns
    -------
    (guide, params, losses) tuple where losses is a list of ELBO values.
    """
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
