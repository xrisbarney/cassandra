"""Generate finite numerical and known-parameter simulation audits for Stage 7."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import nbinom

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.components import kernel_covariance
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for width in np.arange(.25, 18.01, .25):
        ref = expit((width - np.abs(np.arange(-300, 301))) / 2)
        ref /= np.linalg.norm(ref)
        reference = np.array([np.dot(ref[:len(ref)-d], ref[d:]) for d in range(121)])
        actual = kernel_covariance(width, np.arange(121))
        error = np.abs(actual - reference)
        rows.append({'width': float(width), 'max_covariance_error': float(error.max()),
                     'worst_lag': int(error.argmax()), 'lag12_error': float(error[12])})
    rng = np.random.default_rng(7100)
    calibration = []
    for mean in [.1, 1., 20., 100.]:
        for size in [.5, 2., 12.]:
            count = rng.negative_binomial(size, size / (size + mean), size=50000)
            cdf = nbinom.cdf(count, size, size / (size + mean))
            left = nbinom.cdf(count - 1, size, size / (size + mean))
            pit = left + rng.random(len(count)) * (cdf - left)
            record = {'mean': mean, 'dispersion': size, 'replications': len(count),
                      'sample_mean': float(count.mean()),
                      'pit_cdf_max_error': float(np.max(np.abs(
                          np.sort(pit) - (np.arange(len(pit)) + .5) / len(pit)
                      )))}
            for level in [.5, .9]:
                tail = (1 - level) / 2
                lo, hi = nbinom.ppf([tail, 1-tail], size, size / (size + mean))
                actual = nbinom.cdf(hi, size, size/(size+mean))
                actual -= nbinom.cdf(lo-1, size, size/(size+mean))
                record[f'coverage_{int(level*100)}'] = float(((count>=lo)&(count<=hi)).mean())
                record[f'exact_discrete_coverage_{int(level*100)}'] = float(actual)
            calibration.append(record)
    write_json(args.output/'numerics.json', {
        'kernel': rows, 'count_simulation': calibration,
        'interpretation': 'Finite grid and known-parameter simulation, not posterior SBC.',
        'source_sha256': file_hash(Path(__file__)),
        'held_out_scoring_enabled': False,
    })
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
