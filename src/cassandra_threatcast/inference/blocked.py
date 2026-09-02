"""Blocked NUTS/FFBS inference prescribed by paper Steps 21--24."""
from __future__ import annotations

from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS

from cassandra_threatcast.inference.ffbs import ffbs
from cassandra_threatcast.model.paper_exact import paper_model, regime_log_potentials


def _one_continuous_block(
    data: dict,
    config: dict,
    z_path: np.ndarray,
    seed: int,
    warmup: int,
    draws: int,
) -> dict[str, np.ndarray]:
    """Sample the smooth conditional posterior p(continuous | z, data)."""
    conditioned = partial(paper_model, z_path=jnp.asarray(z_path))
    cfg = config.get("mcmc", {})
    kernel = NUTS(
        conditioned,
        target_accept_prob=float(cfg.get("target_accept_prob", 0.9)),
        max_tree_depth=int(cfg.get("max_tree_depth", 10)),
    )
    sampler = MCMC(kernel, num_warmup=warmup, num_samples=draws, progress_bar=False)
    sampler.run(jax.random.PRNGKey(seed), data=data, config=config)
    return {name: np.asarray(value) for name, value in sampler.get_samples().items()}


def run_blocked_nuts_ffbs(data: dict, config: dict) -> az.InferenceData:
    """Alternate NUTS continuous blocks with exact FFBS regime-path draws.

    ``mcmc.num_samples`` is the total number of retained draws, divided among
    ``gibbs_blocks`` alternating updates. This avoids restarting NUTS once per
    draw while preserving the paper's block order.
    """
    mcfg = config.get("mcmc", {})
    T = int(np.shape(data["N"])[1])
    R = int(config.get("model", config)["R"])
    saved = int(mcfg.get("num_samples", 1000))
    burn_blocks = int(mcfg.get("gibbs_warmup_blocks", 2))
    gibbs_blocks = int(mcfg.get("gibbs_blocks", min(20, saved)))
    block_draws = int(mcfg.get("continuous_draws_per_block", int(np.ceil(saved / gibbs_blocks))))
    first_warmup = int(mcfg.get("num_warmup", 1000))
    later_warmup = int(mcfg.get("block_warmup", max(25, first_warmup // 10)))
    seed = int(mcfg.get("seed", 0))

    # Spread the initial path across regimes so every conditional block starts
    # with valid support, while FFBS determines all later paths from the data.
    z_path = np.arange(T, dtype=int) % R
    retained: dict[str, list[np.ndarray]] = {}
    retained_z: list[np.ndarray] = []
    total_blocks = burn_blocks + gibbs_blocks

    for block in range(total_blocks):
        conditional_z = z_path.copy()
        samples = _one_continuous_block(
            data,
            config,
            z_path,
            seed + 2 * block,
            first_warmup if block == 0 else later_warmup,
            block_draws,
        )
        draw = {name: value[-1] for name, value in samples.items()}
        potentials = regime_log_potentials(draw, data)
        z_path = ffbs(
            potentials,
            np.asarray(draw["Pi"]),
            np.ones(R) / R,
            n_samples=1,
            seed=seed + 2 * block + 1,
        )[0]

        if block >= burn_blocks:
            room = saved - len(retained_z)
            keep = min(block_draws, room)
            for sample_idx in range(keep):
                for name, value in samples.items():
                    retained.setdefault(name, []).append(value[sample_idx])
                retained_z.append(conditional_z.copy())

        print(f"      blocked NUTS/FFBS {block + 1}/{total_blocks}", flush=True)

    posterior = {name: np.stack(values)[None, ...] for name, values in retained.items()}
    posterior["z_t"] = np.stack(retained_z)[None, ...]
    # ArviZ 1.x accepts a group mapping rather than keyword group arguments.
    return az.from_dict({"posterior": posterior})


def run_blocked_vi_ffbs(data: dict, config: dict) -> az.InferenceData:
    """Scalable Step-24 alternative: alternate variational blocks with FFBS."""
    from cassandra_threatcast.inference.vi import train_vi, vi_predictive_samples

    T = int(np.shape(data["N"])[1])
    R = int(config.get("model", config)["R"])
    mcfg, scfg = config.get("mcmc", {}), config.get("svi", {})
    n_samples = int(mcfg.get("num_samples", 1000))
    n_blocks = int(scfg.get("vi_gibbs_blocks", 2))
    seed = int(scfg.get("seed", mcfg.get("seed", 0)))
    z_path = np.arange(T, dtype=int) % R
    samples = None
    conditional_z = z_path.copy()

    for block in range(n_blocks):
        conditional_z = z_path.copy()
        conditioned = partial(paper_model, z_path=jnp.asarray(conditional_z))
        guide, params, _ = train_vi(
            conditioned,
            data,
            config,
            num_steps=int(scfg.get("num_steps", 30000)),
            learning_rate=float(scfg.get("learning_rate", 1e-3)),
            seed=seed + 2 * block,
        )
        samples = vi_predictive_samples(
            guide, params, conditioned, data, config=config,
            n_samples=n_samples, seed=seed + 2 * block + 1,
        )
        draw = {name: np.asarray(value[-1]) for name, value in samples.items()}
        potentials = regime_log_potentials(draw, data)
        z_path = ffbs(
            potentials, np.asarray(draw["Pi"]), np.ones(R) / R,
            n_samples=1, seed=seed + 2 * block + 2,
        )[0]
        print(f"      variational/FFBS block {block + 1}/{n_blocks}", flush=True)

    posterior = {
        name: np.asarray(value)[None, ...]
        for name, value in samples.items()
        if name not in {"N_obs", "E_obs", "B_obs", "D_obs"}
    }
    posterior["z_t"] = np.repeat(conditional_z[None, None, :], n_samples, axis=1)
    return az.from_dict({"posterior": posterior})
