"Forward-Filtering Backward-Sampling (FFBS) for discrete Markov regime paths."
from __future__ import annotations

import numpy as np


def forward_filter(
    log_likelihoods: np.ndarray,  # (T, R) log p(obs_t | z_t = r)
    Pi: np.ndarray,               # (R, R) transition matrix
    pi0: np.ndarray,              # (R,) initial distribution
) -> tuple[np.ndarray, np.ndarray]:
    "Hamilton filter forward pass."
    T, R = log_likelihoods.shape
    filtered_probs = np.zeros((T, R))
    predicted_probs = np.zeros((T, R))

    # t = 0: predicted = pi0
    pred_t = pi0.copy()
    predicted_probs[0] = pred_t

    for t in range(T):
        if t > 0:
            # Predict: marginalize over previous regime
            pred_t = filtered_probs[t - 1] @ Pi  # (R,)
            predicted_probs[t] = pred_t

        # Update: multiply by likelihood
        log_lik_t = log_likelihoods[t]  # (R,)
        # Numerically stable: subtract max before exp
        log_joint = np.log(pred_t + 1e-300) + log_lik_t
        log_joint -= log_joint.max()
        joint = np.exp(log_joint)
        norm = joint.sum()
        if norm < 1e-300:
            # Degenerate: fall back to uniform
            filtered_probs[t] = np.ones(R) / R
        else:
            filtered_probs[t] = joint / norm

    return filtered_probs, predicted_probs


def backward_sample(
    filtered_probs: np.ndarray,  # (T, R)
    Pi: np.ndarray,              # (R, R)
    rng: np.random.Generator,
) -> np.ndarray:
    "Stochastic backward pass. Samples z_T, z_{T-1}, …, z_1 sequentially."
    T, R = filtered_probs.shape
    z_path = np.empty(T, dtype=int)

    # Sample z_T from the last filtered distribution
    probs_T = filtered_probs[T - 1]
    probs_T = probs_T / probs_T.sum()
    z_path[T - 1] = rng.choice(R, p=probs_T)

    # Backward pass: t = T-2 down to 0
    for t in range(T - 2, -1, -1):
        z_next = z_path[t + 1]
        # Backward weights: P(z_t | y_{1:t}) * P(z_{t+1} | z_t)
        back_weights = filtered_probs[t] * Pi[:, z_next]  # (R,)
        total = back_weights.sum()
        if total < 1e-300:
            back_weights = np.ones(R) / R
        else:
            back_weights /= total
        z_path[t] = rng.choice(R, p=back_weights)

    return z_path


def ffbs(
    log_likelihoods: np.ndarray,  # (T, R)
    Pi: np.ndarray,               # (R, R)
    pi0: np.ndarray,              # (R,)
    n_samples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    "Combined Forward-Filtering Backward-Sampling."
    rng = np.random.default_rng(seed=seed)
    filtered_probs, _ = forward_filter(log_likelihoods, Pi, pi0)

    T = log_likelihoods.shape[0]
    regime_paths = np.empty((n_samples, T), dtype=int)

    for i in range(n_samples):
        regime_paths[i] = backward_sample(filtered_probs, Pi, rng)

    return regime_paths
