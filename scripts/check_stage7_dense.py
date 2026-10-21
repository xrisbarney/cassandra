"""Check the explicitly experimental dense-metric configuration on a known Gaussian."""

import argparse
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from pymc.step_methods.hmc.quadpotential import QuadPotentialFullAdapt

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    k = 8
    mean = np.linspace(-1, 1, k)
    covariance = .1 * np.eye(k) + .9 * np.ones((k, k))
    with pm.Model(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pm.MvNormal('x', mean, cov=covariance, shape=k)
        potential = QuadPotentialFullAdapt(
            k, mean, np.eye(k), initial_weight=10, adaptation_window=max(101, 2*k),
            update_window=10,
        )
        step = pm.NUTS(potential=potential, target_accept=.95, max_treedepth=12,
                       compile_kwargs={'mode':'NUMBA'})
        idata = pm.sample(draws=1000, tune=1000, chains=4, cores=1, step=step,
                          random_seed=7100, progressbar=False)
    diagnostic, _ = diagnostic_report(idata, 12)
    draws = idata.posterior.x.values.reshape(-1, k)
    mcse = az.mcse(idata).x.values
    mean_error = float(np.max(np.abs(draws.mean(axis=0)-mean) / mcse))
    covariance_error = float(np.max(np.abs(np.cov(draws.T)-covariance)))
    passed = diagnostic['accepted'] and mean_error < 5 and covariance_error < .08
    write_json(args.output, {
        'purpose': 'Sampler configuration check, not state-space recovery or performance.',
        'diagnostics': diagnostic, 'max_mean_error_in_mcse': mean_error,
        'max_absolute_covariance_error': covariance_error,
        'warnings': list(dict.fromkeys(str(w.message) for w in caught)),
        'status': 'passed' if passed else 'failed',
        'source_sha256': file_hash(Path(__file__)),
    })
    if not passed:
        raise RuntimeError('Dense-metric known-Gaussian validation failed.')


if __name__ == '__main__':
    main()
