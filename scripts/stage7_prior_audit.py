"""Finite prior predictive stress audit; no clipping or rejection of large log rates."""

import argparse
from pathlib import Path

import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rng = np.random.default_rng(7100)
    draws, k, rank = 20000, 4, 1
    group = rng.normal(0, 3, size=(draws, 1))
    sd = np.abs(rng.normal(0, 2, size=(draws, 1)))
    m = group + sd * rng.normal(size=(draws, k))
    phi = rng.uniform(-1, 1, size=(draws, rank))
    stationary_f = rng.normal(size=(draws, rank)) / np.sqrt(1 - phi**2)
    loading = rng.normal(size=(draws, k, rank))
    loading[:, 0, 0] = np.abs(loading[:, 0, 0])
    eta = m + np.einsum('skr,sr->sk', loading, stationary_f)
    psi = rng.lognormal(np.log(10), 1, size=(draws, k))
    # Report operational simulation overflow, preserving its index and log rate.
    failures, max_count = [], 0
    for i in range(draws):
        for j in range(k):
            try:
                with np.errstate(over='raise', under='ignore'):
                    mu = np.exp(eta[i, j])
                    count = rng.negative_binomial(psi[i, j], psi[i, j] / (psi[i, j] + mu))
                max_count = max(max_count, int(count))
            except (ValueError, FloatingPointError) as error:
                failures.append({'draw': i, 'category': j, 'log_rate': float(eta[i, j]),
                                 'reason': type(error).__name__})
    write_json(args.output, {
        'draws': draws, 'categories': k, 'factors': rank, 'seed': 7100,
        'prior': 'Frozen stationary factor prior, counts only, no iid residual, e=0.',
        'log_rate_quantiles': {str(q): float(np.quantile(eta, q))
                               for q in [0, .01, .5, .99, 1]},
        'log_rate_above_20': int((eta > 20).sum()),
        'count_simulation_failures': failures, 'largest_successful_count': max_count,
        'source_sha256': file_hash(Path(__file__)),
        'interpretation': 'Finite stress sample, not a bound or calibrated prediction.',
    })
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
