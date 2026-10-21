"""Describe covariance recovery and prior contraction for admitted synthetic fits.

This is descriptive analysis of the already recorded design. It does not select
fits, introduce acceptance thresholds, or change any posterior or forecast.
"""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import lognorm, truncnorm

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic_recovery import (
    interval_metrics,
    posterior_array,
    reconstructed_loading,
)
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID, amended
from cyber_threat_modelling.provenance import file_hash


def covariance(loading, phi, a, width, tau, phi_u, residual, lag, revised=False):
    """Stationary log-state covariance conditional on the regime and effort."""
    size, k, _ = loading.shape
    factor_var = phi**lag if revised else phi**lag / (1-phi**2)
    result = np.einsum('skr,sr,slr->skl', loading, factor_var, loading)
    if a is not None:
        from scipy.special import expit

        raw = expit((width[:, None] - np.abs(np.arange(61) - 30)) / 2)
        weights = raw / np.linalg.norm(raw, axis=1)[:, None]
        cov_g = np.sum(weights[:, :61-lag] * weights[:, lag:], axis=1)
        result += a[:, :, None] * a[:, None, :] * cov_g[:, None, None]
    if residual == 'ar':
        diagonal = tau**2 * phi_u**lag / (1 if revised else 1-phi_u**2)
    elif residual == 'iid' and lag == 0:
        diagonal = tau**2
    else:
        diagonal = np.zeros((size, k))
    result[:, np.arange(k), np.arange(k)] += diagonal
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    contractions, covariances, details = [], [], []
    rng = np.random.default_rng(20261021)
    width_prior_sd = np.sqrt(lognorm.expect(
        lambda x: x*x, args=(.5,), scale=12, lb=0, ub=18, conditional=True
    ) - lognorm.expect(lambda x: x, args=(.5,), scale=12, lb=0, ub=18,
                       conditional=True)**2)
    for case in audit['runs']:
        folder = Path(case['directory'])
        if file_hash(folder/'result.json') != case['result_sha256']:
            raise ValueError(f'Changed result: {folder}')
        result = json.loads((folder/'result.json').read_text())
        for name, digest in result['artifacts'].items():
            if file_hash(folder/name) != digest:
                raise ValueError(f'Changed artifact: {folder/name}')
        if case['status'] != 'accepted':
            continue
        options = case['options']
        revised = amended(options)
        prefix = {'scenario': case['scenario'], 'seed': case['seed']}
        post = az.from_netcdf(folder/f"posterior_{result['attempts'][-1]['retry']}.nc").posterior
        with np.load(folder/'truth.npz') as loaded:
            truth = dict(loaded)
        get = lambda name: posterior_array(post, name)  # noqa: E731
        samples = len(get('psi'))
        k, rank = options['categories'], options['factors']
        loading = reconstructed_loading(post, 'f', k, rank, np.arange(samples)) if rank else (
            np.zeros((samples, k, 0))
        )
        phi = get('f_phi') if rank else np.zeros((samples, 0))
        a, width = (get('a'), get('width')) if options['kernel'] else (None, None)
        tau = get('tau') if options['residual'] != 'none' else np.zeros((samples, k))
        phi_u = get('phi_u') if options['residual'] == 'ar' else np.zeros((samples, k))
        for lag in (0, 1, 6, 12):
            draws = covariance(loading, phi, a, width, tau, phi_u, options['residual'], lag,
                               revised)
            target = covariance(
                truth['loading'][None], truth['phi'][None],
                truth['a'][None] if options['kernel'] else None,
                truth['width'][None] if options['kernel'] else None,
                truth['tau'][None], truth['phi_u'][None], options['residual'], lag, revised,
            )[0]
            covariances.append({**prefix, 'lag': lag, **interval_metrics(draws, target),
                                'truth_frobenius_norm': float(np.linalg.norm(target))})
        priors = {'m': np.sqrt(13), 'n': np.sqrt(2), 'f_phi': np.sqrt(1/3),
                  'h_phi': np.sqrt(1/3), 'phi_u': np.sqrt(1/3),
                  'tau': np.sqrt(1-2/np.pi), 'width': width_prior_sd}
        if revised:
            priors.update(m=np.sqrt(9 + truncnorm.moment(2, 0, 1.5, scale=2)),
                          n=np.sqrt(1 + truncnorm.moment(2, 0, 2)),
                          tau=np.sqrt(truncnorm.var(0, 1)))
        settings = (options.get('level_settings')
                    if options.get('prior_id') == CURRENT_PRIOR_ID else None)
        multiplier = settings['amplitude_multiplier'] if settings else 1.
        if settings:
            scale, bound = settings['scale_sd'], settings['scale_upper']
            priors['m'] = np.sqrt(settings['hyper_sd']**2
                                   + truncnorm.moment(2, 0, bound/scale, scale=scale))
            priors['tau'] *= multiplier
        for name, sd in priors.items():
            if name in post:
                values = get(name).reshape(samples, -1)
                for j, actual in enumerate(values.std(axis=0)):
                    marginal_sd = 1. if (settings and name == 'm'
                                          and j == settings['fallback_index']) else sd
                    contractions.append({**prefix, 'parameter': name, 'index': j,
                                         'prior_sd': marginal_sd, 'posterior_sd': actual,
                                         'sd_ratio': actual/marginal_sd})
        for j, actual in enumerate(np.log(get('psi')).std(axis=0)):
            contractions.append({**prefix, 'parameter': 'log_psi', 'index': j,
                                 'prior_sd': 1., 'posterior_sd': actual, 'sd_ratio': actual})
        recovery = json.loads((folder/'recovery.json').read_text())
        extra = {**prefix}
        if revised:
            bounds = {'m_sd': 3, 'n_sd': 2, 'tau': 1, 'a': 1}
            if settings:
                bounds.update(m_sd=settings['scale_upper'], tau=multiplier, a=multiplier)
            for prefix_name, factor_rank in [('f', rank), ('h', options['severity_factors'])]:
                if factor_rank:
                    bounds[prefix_name+'_diag'] = 2*multiplier/np.sqrt(factor_rank)
                    bounds[prefix_name+'_free'] = 2*multiplier/np.sqrt(factor_rank)
            extra['upper_support_edge_probability'] = {
                name: np.mean(np.abs(get(name)) > .95*bound, axis=0).tolist()
                for name, bound in bounds.items() if name in post
            }
        for key in ('tau_logpsi_correlation', 'regimes', 'fallback_check'):
            if recovery.get(key) is not None:
                extra[key] = recovery[key]
        if options['kernel']:
            # The first loading is positive but squaring removes its sign;
            # hence its prior norm is a half-Normal scale times a chi norm.
            norm_prior = abs(rng.normal(0, .5, 200000)) * np.sqrt(rng.chisquare(k, 200000))
            if revised:
                scale = abs(rng.normal(0, .5*multiplier, (200000, 1)))
                values = truncnorm.rvs(-multiplier/scale, multiplier/scale, scale=scale,
                                       size=(200000, k), random_state=rng)
                norm_prior = np.linalg.norm(values, axis=1)
            norm = get('a_norm')
            extra['kernel'] = {
                'prior_norm_mean': float(norm_prior.mean()),
                'prior_norm_q90': float(np.quantile(norm_prior, .9)),
                'posterior_norm_mean': float(norm.mean()),
                'posterior_norm_q90': float(np.quantile(norm, .9)),
                'posterior_width_sd': float(width.std()), 'prior_width_sd': width_prior_sd,
                'width_norm_correlation': float(np.corrcoef(width, norm)[0, 1]),
                'truth_norm': float(np.linalg.norm(truth['a'])),
                'note': 'Width is unidentified at zero loading; width recovery is not required.',
            }
        details.append(extra)
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(contractions).to_csv(args.output/'contraction.csv', index=False)
    pd.DataFrame(covariances).to_csv(args.output/'covariance.csv', index=False)
    write_json(args.output/'details.json', {
        'cases': details, 'audit_sha256': file_hash(args.audit),
        'analysis_source_sha256': file_hash(Path(__file__)),
        'interpretation': 'Descriptive accepted-fit summaries; no new admission thresholds.',
        'covariance_target': 'Stationary log-state covariance conditional on regime and effort.',
        'contraction': 'Posterior SD divided by the matching recorded marginal prior SD.',
        'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
    })


if __name__ == '__main__':
    main()
