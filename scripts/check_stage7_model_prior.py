"""Longitudinal prior check through the amended model's actual random variables."""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pymc as pm

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import PRIOR_ID, verify_activation
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash, git_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    activation = verify_activation(root)
    source = {'source_revision': git_revision(root), 'source_sha256': file_hash(Path(__file__)),
              'model_sha256': file_hash(
                  root/'src/cyber_threat_modelling/models/synthetic_state_space.py'),
              'prior_activation': activation, 'real_data_read': False}
    write_json(args.output/'run.json', source)
    results = []
    for index, full in enumerate((False, True)):
        data, truth, options = generate('representative', 7101)
        options.update(prior_id=PRIOR_ID, factor_parameterization='innovations')
        if full:
            options.update(kernel=True, residual='iid', regimes=3)
        model = build_model(data, options, truth)
        names = ['eta_by_regime', 'psi', 'f_phi', 'f_states']
        if full:
            names += ['transition', 'regime_increments', 'a', 'tau']
        with model, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prior = pm.sample_prior_predictive(
                draws=1000, var_names=names, random_seed=72101+index,
                return_inferencedata=False, compile_kwargs={'mode': 'FAST_COMPILE'},
            )
        rng = np.random.default_rng(72201+index)
        rates = prior['eta_by_regime']
        draws, months, regimes, k = rates.shape
        z = rng.integers(regimes, size=draws)
        eta = np.empty((draws, months, k))
        for t in range(months):
            if t and full:
                cdf = np.cumsum(prior['transition'][np.arange(draws), z], axis=1)
                z = (rng.random(draws)[:, None] > cdf).sum(axis=1)
            eta[:, t] = rates[np.arange(draws), t, z]
        failures, count_values = [], []
        for draw in range(draws):
            try:
                with np.errstate(over='raise', invalid='raise'):
                    mu = np.exp(eta[draw])
                    psi = prior['psi'][draw]
                    counts = rng.negative_binomial(psi, psi/(psi+mu))
                count_values.append(counts)
            except (ValueError, FloatingPointError) as error:
                failures.append({'draw': draw, 'type': type(error).__name__,
                                 'max_log_rate': float(eta[draw].max())})
        artifact = args.output/('full.npz' if full else 'pilot.npz')
        np.savez_compressed(artifact, **prior)
        phi = prior['f_phi']
        states = prior['f_states']
        innovations = (states[:, 1:]-phi[:, None]*states[:, :-1])/np.sqrt(1-phi[:, None]**2)
        quantiles = [0, .01, .5, .99, 1]
        results.append({
            'case': 'full' if full else 'pilot', 'options': options, 'draws': draws,
            'regimes': 'initial uniform, subsequent sampled transition rows',
            'effort': 0, 'failures': failures, 'log_rate_above_20': int((eta > 20).sum()),
            'log_rate_quantiles': dict(zip(map(str, quantiles), np.quantile(eta, quantiles))),
            'count_quantiles': dict(zip(map(str, quantiles),
                                       np.quantile(np.stack(count_values), quantiles))),
            'initial_factor_variance': np.var(states[:, 0], axis=0).tolist(),
            'standardized_innovation_mean': np.mean(innovations, axis=(0, 1)).tolist(),
            'standardized_innovation_variance': np.var(innovations, axis=(0, 1)).tolist(),
            'warnings': sorted({str(w.message) for w in caught}),
            'artifact_path': artifact.as_posix(), 'artifact_sha256': file_hash(artifact),
        })
    write_json(args.output/'audit.json', {
        **source, 'cases': results,
        'interpretation': 'Prior simulation only; not posterior recovery or SBC. '
                          'Potential likelihoods are intentionally ignored.',
    })
    print(args.output/'audit.json', flush=True)


if __name__ == '__main__':
    main()
