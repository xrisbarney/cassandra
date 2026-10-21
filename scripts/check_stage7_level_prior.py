"""Training-only plausibility audit of the actual v2 model, never a posterior fit."""

import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pymc as pm

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.components import fitted_effort
from cyber_threat_modelling.models.stage7_priors import (
    CANDIDATES,
    CURRENT_PRIOR_ID,
    level_settings,
    verify_activation,
)
from cyber_threat_modelling.models.synthetic_state_space import build_model
from cyber_threat_modelling.provenance import file_hash, git_revision


def metrics(counts):
    return {'median': float(np.median(counts)), 'q999': float(np.quantile(counts, .999)),
            'p_zero': float(np.mean(counts == 0)), 'maximum': int(counts.max())}


def uncertainty(counts, rng):
    # An entire trajectory/parameter draw is the resampling unit. Reweight its
    # empirical histogram; this is exactly the resampled empirical distribution,
    # without sorting millions of repeated counts 200 times.
    flat = counts.reshape(len(counts), -1)
    values, inverse = np.unique(flat.ravel(), return_inverse=True)
    from scipy.sparse import csr_matrix
    rows = np.repeat(np.arange(len(flat)), flat.shape[1])
    hist = csr_matrix((np.ones(flat.size, dtype=np.int32), (rows, inverse)),
                      shape=(len(flat), len(values)))
    output = []
    for _ in range(200):
        weights = np.bincount(rng.integers(len(flat), size=len(flat)), minlength=len(flat))
        freq = np.asarray(weights @ hist).ravel()
        cdf = freq.cumsum()
        def quantile(q):
            position = q*(cdf[-1]-1)
            lo, hi = int(np.floor(position)), int(np.ceil(position))
            a, b = values[np.searchsorted(cdf, [lo+1, hi+1])]
            return float(a+(position-lo)*(b-a))
        output.append([quantile(.5), quantile(.999),
                       float(freq[0]/cdf[-1]) if values[0] == 0 else 0.])
    return dict(zip(['median', 'q999', 'p_zero'],
                    np.quantile(output, [.025, .975], axis=0).T.tolist()))


def gate(stats, intervals, failures):
    limits = {'median': (30, 175), 'q999': (0, 50000), 'p_zero': (0, .05)}
    passed = not failures and all(lo <= stats[key] <= hi for key, (lo, hi) in limits.items())
    unresolved = passed and any(intervals[key][0] < lo or intervals[key][1] > hi
                                for key, (lo, hi) in limits.items())
    return 'unresolved' if unresolved else 'pass' if passed else 'fail'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, choices=[2000, 8000], default=2000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    activation = verify_activation(root, CURRENT_PRIOR_ID)
    args.output.mkdir(parents=True, exist_ok=False)
    panel = root/'data/processed/stage5-20261020/topic_month.parquet'
    prefix = pq.read_table(panel, columns=['month', 'category', 'N'],
                           filters=[('month', '>=', '2010-01'), ('month', '<', '2019-01')])
    table = prefix.to_pandas().pivot(index='month', columns='category', values='N')
    if table.shape != (108, 10) or list(table.columns) != list(range(1, 11)):
        raise ValueError('Unexpected initial-window panel.')
    counts = table.to_numpy(dtype=np.int64)
    effort, effort_fit = fitted_effort(counts)
    data, _, options = generate('representative', 7101)
    data.update(N=counts, mask_N=np.ones_like(counts, dtype=bool), effort=np.zeros(108))
    options.update(prior_id=CURRENT_PRIOR_ID, factor_parameterization='innovations',
                   fallback_index=9)
    np.savez_compressed(args.output/'training_prefix.npz', N=counts, effort=effort)
    source = {'prior_activation': activation, 'source_control': git_revision(root),
              'source_sha256': {p.relative_to(root).as_posix(): file_hash(p)
                               for p in [Path(__file__), *list((root/'src').rglob('*.py'))]},
              'real_data_read': '2010-01 through 2018-12 counts only',
              'held_out_scoring_enabled': False, 'effort_fit': effort_fit,
              'training_prefix_sha256': file_hash(args.output/'training_prefix.npz'),
              'initial_category_means': counts.mean(axis=0).tolist()}
    write_json(args.output/'run.json', source)
    cases = []
    for case_index, (candidate, full) in enumerate(
            [(name, False) for name in CANDIDATES]+[('primary', True)]):
        config = dict(options, prior_candidate=candidate, kernel=full,
                      residual='iid' if full else 'none', regimes=3 if full else 1)
        model = build_model(data, config)
        names = ['eta_by_regime', 'psi', 'm', 'm_sd', 'f_phi', 'f_states']
        if full:
            names += ['transition', 'regime_increments']
        with model, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prior = pm.sample_prior_predictive(args.draws, var_names=names,
                                               random_seed=73101+case_index,
                                               return_inferencedata=False,
                                               compile_kwargs={'mode': 'FAST_COMPILE'})
        rng = np.random.default_rng(73201+case_index)
        rates = prior['eta_by_regime']
        draws, months, regimes, _ = rates.shape
        z = rng.integers(regimes, size=draws)
        eta = np.empty((draws, months, 10))
        occupancy = np.zeros(regimes, dtype=int)
        for t in range(months):
            if t and full:
                cdf = np.cumsum(prior['transition'][np.arange(draws), z], axis=1)
                z = np.minimum((rng.random(draws)[:, None] > cdf).sum(axis=1), regimes-1)
            occupancy += np.bincount(z, minlength=regimes)
            eta[:, t] = rates[np.arange(draws), t, z]
        record = {'candidate': candidate, 'full': full, 'draws': draws,
                  'settings': level_settings(data, config),
                  'regime_occupancy': (occupancy/occupancy.sum()).tolist(),
                  'warnings': sorted({str(w.message) for w in caught})}
        artifact = args.output/f'{candidate}-{full}.npz'
        np.savez_compressed(artifact, **prior)
        record['prior_artifact_sha256'] = file_hash(artifact)
        for label, log_mu in [('reference', eta), ('longitudinal', eta+effort[None, :, None]),
                              *([('stress_envelope', rates[:, :, -1]+3)] if full else [])]:
            values, failures = [], []
            for draw in range(draws):
                try:
                    with np.errstate(over='raise', invalid='raise'):
                        mu = np.exp(log_mu[draw])
                        alpha = prior['psi'][draw]
                        values.append(rng.negative_binomial(alpha, alpha/(alpha+mu)))
                except (ValueError, FloatingPointError) as error:
                    failures.append({'draw': draw, 'type': type(error).__name__})
            if failures:
                record[label] = {'failures': failures, 'gate': 'fail'}
                continue
            values = np.stack(values)
            known = values[:, :, :9]
            stats = metrics(known)
            intervals = uncertainty(known, rng) if label == 'reference' and not full else None
            status = gate(stats, intervals, failures) if intervals else 'diagnostic_only'
            if label == 'longitudinal' and not full:
                status = 'pass' if stats['q999'] <= 50000*np.exp(effort.max()) else 'fail'
            record[label] = {'known': stats, 'fallback': metrics(values[:, :, 9]),
                             'bootstrap_95': intervals, 'failures': failures, 'gate': status}
        cases.append(record)
        write_json(args.output/'audit.json', {**source, 'cases': cases, 'complete': False})
        print({k: record[k] for k in ['candidate', 'full', 'reference']}, flush=True)
        del prior, model, rates, eta, values, known
        gc.collect()
    primary = cases[0]
    passed = all(primary[name]['gate'] == 'pass' for name in ['reference', 'longitudinal'])
    write_json(args.output/'audit.json', {**source, 'cases': cases, 'complete': True,
                                        'primary_passed': passed,
                                        'interpretation': 'Data-informed prior plausibility; '
                                        'not posterior recovery or SBC.'})


if __name__ == '__main__':
    main()
