"Damage functions and Leontief input-output propagation for cyber loss estimation."
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from scipy.optimize import minimize


# Reference cyber events used to calibrate damage functions
DEFAULT_REFERENCE_EVENTS: list[dict] = [
    {
        "name": "NotPetya",
        "year": 2017,
        "sector_idx": 4,  # Manufacturing (global supply chain disruption)
        "loss_lo": 8.0e9,
        "loss_hi": 12.0e9,
        "x_s_fraction": 0.002,  # shock as fraction of sector gross output
        "description": "Wiper malware; $10B+ in global losses; Maersk, Merck, FedEx",
    },
    {
        "name": "Colonial_Pipeline",
        "year": 2021,
        "sector_idx": 6,  # Transportation & warehousing
        "loss_lo": 3.0e9,
        "loss_hi": 6.0e9,
        "x_s_fraction": 0.0015,
        "description": "Ransomware; ~$4.4B direct + indirect; US East Coast fuel disruption",
    },
    {
        "name": "MOVEit",
        "year": 2023,
        "sector_idx": 9,  # Professional & business services
        "loss_lo": 7.0e9,
        "loss_hi": 13.0e9,
        "x_s_fraction": 0.001,
        "description": "SQL injection in MFT software; ~$9.9B; 2,600+ organisations",
    },
    {
        "name": "Change_Healthcare",
        "year": 2024,
        "sector_idx": 10,  # Health care & social assistance
        "loss_lo": 18.0e9,
        "loss_hi": 26.0e9,
        "x_s_fraction": 0.003,
        "description": "Ransomware; ~$22B; US healthcare claims processing paralysed",
    },
]


# Dataclass for per-sector damage function parameters

@dataclass
class DamageFunctionParams:
    "Per-sector parameters for the damage function phi_s(load)."
    shape: np.ndarray      # (S,) sigmoid steepness
    scale: np.ndarray      # (S,) midpoint of sigmoid (fraction of capacity)
    max_damage: np.ndarray # (S,) maximum fraction of output that can be disrupted


# Damage function

def damage_function(
    shock_load: jnp.ndarray,       # (S,)
    params: DamageFunctionParams,
) -> jnp.ndarray:
    "Maps per-sector shock loads to fraction of output disrupted."
    shape = jnp.asarray(params.shape)      # (S,)
    scale = jnp.asarray(params.scale)      # (S,)
    max_damage = jnp.asarray(params.max_damage)  # (S,)

    s = jax_sigmoid((shock_load - scale) * shape)
    s0 = jax_sigmoid((0.0 - scale) * shape)   # sigmoid value at zero load
    g_s = max_damage * (s - s0) / jnp.clip(1.0 - s0, 1e-8, None)
    return g_s


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    "Numerically stable sigmoid."
    return jnp.where(x >= 0, 1.0 / (1.0 + jnp.exp(-x)), jnp.exp(x) / (1.0 + jnp.exp(x)))


# Leontief propagation

def leontief_propagation(
    g_s: jnp.ndarray,       # (S,) fraction of output disrupted per sector
    x_s: jnp.ndarray,       # (S,) gross output by sector
    Lambda_L: jnp.ndarray,  # (S, S) Leontief inverse
) -> tuple[jnp.ndarray, jnp.ndarray]:
    "Compute direct and propagated losses."
    d_s = g_s * x_s        # (S,)
    ell = Lambda_L @ d_s   # (S,)
    return d_s, ell


# Calibration

def calibrate_damage_functions(
    reference_events: list[dict],
    Lambda_L: np.ndarray,
    x_s: np.ndarray,
    S: int,
    num_samples: int = 1000,
) -> DamageFunctionParams:
    "Fit phi_s parameters using documented loss ranges from reference cyber events"
    # Initialise parameters with sensible defaults
    shape_init = np.full(S, 10.0)
    scale_init = np.full(S, 0.01)
    max_damage_init = np.full(S, 0.05)

    def _pack(shape, scale, max_damage):
        return np.concatenate([shape, scale, max_damage])

    def _unpack(theta):
        shape = theta[:S]
        scale = theta[S: 2 * S]
        max_damage = theta[2 * S:]
        return shape, scale, max_damage

    def _objective(theta):
        shape, scale, max_damage = _unpack(theta)
        loss_total = 0.0
        for event in reference_events:
            s_idx = event["sector_idx"]
            frac = event["x_s_fraction"]
            loss_mid = 0.5 * (event["loss_lo"] + event["loss_hi"])

            # Construct shock load: only sector s_idx is shocked
            shock = np.zeros(S)
            shock[s_idx] = frac

            # Sigmoid damage
            logit = (shock - scale) * shape
            g_s = max_damage / (1.0 + np.exp(-logit))

            # Propagate
            d_s = g_s * x_s
            ell = Lambda_L @ d_s
            total_model_loss = float(np.sum(ell))

            # Relative squared error against mid-point loss
            rel_err = (total_model_loss - loss_mid) / (loss_mid + 1e-12)
            loss_total += rel_err ** 2

        # Regularisation: keep parameters near sensible defaults
        loss_total += 0.01 * np.sum((shape - 10.0) ** 2)
        loss_total += 0.01 * np.sum((scale - 0.01) ** 2)
        loss_total += 0.01 * np.sum((max_damage - 0.05) ** 2)
        return loss_total

    theta0 = _pack(shape_init, scale_init, max_damage_init)

    bounds = (
        [(1.0, 100.0)] * S   # shape: positive steepness
        + [(1e-4, 0.5)] * S  # scale: fraction of capacity
        + [(1e-4, 0.5)] * S  # max_damage: at most 50% disruption
    )

    result = minimize(
        _objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )

    shape_fit, scale_fit, max_damage_fit = _unpack(result.x)

    return DamageFunctionParams(
        shape=shape_fit,
        scale=scale_fit,
        max_damage=max_damage_fit,
    )
