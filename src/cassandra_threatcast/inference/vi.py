"Variational inference utilities wrapping NumPyro's AutoLowRankMultivariateNormal."
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import ClippedAdam


class AutoGuide:
    "Wraps NumPyro's AutoLowRankMultivariateNormal with rank from config."

    def __init__(self, model, rank: int = 10) -> None:
        "model : NumPyro model function."
        self.model = model
        self.rank = rank
        self._guide: AutoLowRankMultivariateNormal | None = None

    def get_guide(self) -> AutoLowRankMultivariateNormal:
        "Return (and lazily construct) the AutoLowRankMultivariateNormal guide."
        if self._guide is None:
            self._guide = AutoLowRankMultivariateNormal(self.model, rank=self.rank)
        return self._guide


def train_vi(
    model,
    data: dict,
    config: dict,
    num_steps: int = 30000,
    learning_rate: float = 1e-3,
    seed: int = 0,
    num_particles: int = 4,
) -> tuple:
    "Run SVI training loop."
    model_cfg = config.get("model", {})
    rank = int(model_cfg.get("vi_rank", 10))

    auto = AutoGuide(model, rank=rank)
    guide = auto.get_guide()

    # ClippedAdam with gradient clipping to stabilise early training
    optimizer = ClippedAdam(step_size=learning_rate, clip_norm=10.0)

    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(num_particles=num_particles),
    )

    rng_key = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng_key, data=data, config=config)

    losses_list: list[float] = []

    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, data=data, config=config)
        loss_val = float(loss)
        losses_list.append(loss_val)

        if step % 1000 == 0:
            elbo = -loss_val
            print(f"[VI] step {step:6d} / {num_steps}  ELBO = {elbo:>14.4f}")

    params = svi.get_params(svi_state)
    return guide, params, losses_list


def vi_predictive_samples(
    guide,
    params,
    model,
    data: dict,
    config: dict | None = None,
    n_samples: int = 1000,
    seed: int = 0,
) -> dict:
    "Draw posterior samples from the fitted VI guide."
    config = config if config is not None else {}
    rng_key = jax.random.PRNGKey(seed)

    predictive = Predictive(
        model=guide,
        params=params,
        num_samples=n_samples,
        return_sites=None,  # return all latent sites from the guide
    )
    # Draw samples from the guide (posterior approximation)
    guide_samples = predictive(rng_key, data=data, config=config)

    # Draw from the full model conditioned on guide samples so the returned
    pred_model = Predictive(
        model=model,
        posterior_samples=guide_samples,
        return_sites=None,
    )
    model_samples = pred_model(rng_key, data=data, config=config)

    # Merge: model_samples includes both latent and observed sites
    return {**guide_samples, **model_samples}
