"""Audit a proposed prior without installing it in any inference or forecast model.

Stationary-month simulations, not a longitudinal recovery or SBC experiment.
All numerical failures are retained; outputs and rates are never clipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import truncnorm

from cyber_threat_modelling.provenance import file_hash


def bounded_normal(rng, sd, bound, size, positive=False):
    lower = 0 if positive else -bound
    return truncnorm.rvs(lower / sd, bound / sd, scale=sd,
                         size=size, random_state=rng)


def summaries(values):
    return {str(q): float(np.quantile(values, q))
            for q in (0, .001, .01, .5, .95, .99, .999, 1)}


def count_check(rng, log_mu, psi):
    failures = []
    counts = []
    for index in np.ndindex(log_mu.shape):
        try:
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                mu = np.exp(log_mu[index])
                p = psi[index] / (psi[index] + mu)
                count = rng.negative_binomial(psi[index], p)
            counts.append(int(count))
        except (ValueError, FloatingPointError) as error:
            failures.append({'index': list(index), 'log_mu': float(log_mu[index]),
                             'error': type(error).__name__})
    return {'failures': failures, 'successful_draws': len(counts),
            'count_quantiles': summaries(counts),
            'log_mu_quantiles': summaries(log_mu),
            'log_mu_above_20': int((log_mu > 20).sum())}


def simulate(config, case, rng):
    n, k, rank = config['prior_check_draws_per_case'], case['categories'], case['rank']
    group = rng.normal(config['group_mean_location'], config['group_mean_sd'], (n, 1))
    sd = bounded_normal(rng, config['category_level_scale_hn_sd'],
                        config['category_level_scale_upper'], (n, 1), True)
    levels = group + sd * rng.normal(size=(n, k))
    load_sd = config['loading_normal_sd_numerator'] / np.sqrt(rank)
    load_bound = config['loading_absolute_bound_numerator'] / np.sqrt(rank)
    load = bounded_normal(rng, load_sd, load_bound, (n, k, rank))
    for j in range(rank):
        load[:, j, j] = np.abs(load[:, j, j])
        load[:, j, j+1:] = 0
    # Phi changes serial covariance, but not the revised stationary marginal.
    eta = levels + np.einsum('nkj,nj->nk', load, rng.normal(size=(n, rank)))
    variance = np.sum(load**2, axis=-1)
    if case['iid']:
        tau = bounded_normal(rng, config['iid_scale_hn_sd'],
                             config['iid_scale_upper'], (n, k), True)
        eta += tau * rng.normal(size=(n, k))
        variance += tau**2
    if case['kernel']:
        sa = np.abs(rng.normal(0, config['kernel_hyper_scale_hn_sd'], (n, 1)))
        a = bounded_normal(rng, sa, config['kernel_loading_absolute_bound'], (n, k))
        a[:, 0] = np.abs(a[:, 0])
        eta += a * rng.normal(size=(n, 1))
        variance += a**2
    increments = np.abs(rng.normal(0, config['regime_increment_hn_sd'],
                                   (n, case['regimes'] - 1)))
    shifts = np.column_stack((np.zeros(n), np.cumsum(increments, axis=1)))
    # Deliberate upper-shift envelope, not a draw from the Markov occupancy law.
    eta += shifts[:, -1, None]
    psi = rng.lognormal(config['nb_log_dispersion_mean'],
                        config['nb_log_dispersion_sd'], (n, k))
    result = {'case': case, 'regime_policy': 'highest regime envelope at stationary month',
              'dynamic_variance_max': float(variance.max()),
              'counts': count_check(rng, eta + case['effort'], psi)}
    alpha = rng.normal(-3, 2, (n, k))
    beta = bounded_normal(rng, config['kev_slope_normal_sd'],
                          config['kev_slope_absolute_bound'], (n, k))
    result['kev'] = count_check(rng, alpha + beta * eta, rng.lognormal(0, 1, (n, k)))
    return result


def analytic_checks(config):
    # Compare direct quadrature with the exact independent-level marginal MGF.
    s, upper = config['category_level_scale_hn_sd'], config['category_level_scale_upper']
    normalizer = 2 * ndtr(upper / s) - 1
    checks = []
    for power in (1, 2):
        expectation, error = quad(
            lambda x: np.exp(.5 * power**2 * x**2 - .5 * (x / s)**2)
            * np.sqrt(2 / np.pi) / s / normalizer, 0, upper)
        log_moment = (power * config['group_mean_location']
                      + .5 * power**2 * config['group_mean_sd']**2 + np.log(expectation))
        checks.append({'rate_moment_power': power, 'level_log_moment': float(log_moment),
                       'quadrature_abs_error': error,
                       'finite_variance_bound_log_moment':
                           power * config['group_mean_location']
                           + .5 * power**2 * (config['group_mean_sd']**2 + upper**2)})
    phi = np.array([-.9999, -.9, 0, .65, .9999])
    stationary = (1 - phi**2) / (1 - phi**2)
    # For rank one and power one, the truncated N(0,1) loading density cancels
    # exp(G**2/2) exactly. This independently checks the quadrature mechanism.
    loading_bound = config['loading_absolute_bound_numerator']
    loading_normalizer = 2 * ndtr(loading_bound) - 1
    factor_closed = 2 * loading_bound / (np.sqrt(2 * np.pi) * loading_normalizer)
    factor_quad, _ = quad(lambda x: np.exp(x*x/2) * np.exp(-x*x/2)
                          / (np.sqrt(2*np.pi) * loading_normalizer),
                          -loading_bound, loading_bound)
    assert np.isclose(factor_quad, factor_closed, rtol=1e-12)
    return {'level_moments': checks, 'phi_values': phi.tolist(),
            'rank_one_factor_first_moment_closed_form': factor_closed,
            'rank_one_factor_first_moment_quadrature': factor_quad,
            'stationary_variance_from_transition': stationary.tolist(),
            'conditional_dynamic_variance_upper':
                config['loading_absolute_bound_numerator']**2
                + config['iid_scale_upper']**2 + config['kernel_loading_absolute_bound']**2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.proposal.read_text())
    proposal_hash, script_hash = file_hash(args.proposal), file_hash(Path(__file__))
    rng = np.random.default_rng(config['prior_check_seed'])
    results = [simulate(config, case, rng) for case in config['prior_check_cases']]
    record = {'status': 'draft_prior_audit_not_activation', 'cases': results,
              'analytic_checks': analytic_checks(config),
              'proposal_sha256': proposal_hash, 'script_sha256': script_hash,
              'seed': config['prior_check_seed'], 'real_data_read': False,
              'posterior_fits_run': False,
              'limits': 'Finite simulations do not prove tail bounds or posterior convergence.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for result in results:
        print(result['case']['name'], 'N failures', len(result['counts']['failures']),
              'V failures', len(result['kev']['failures']),
              'N max log mean', result['counts']['log_mu_quantiles']['1'])


if __name__ == '__main__':
    main()
