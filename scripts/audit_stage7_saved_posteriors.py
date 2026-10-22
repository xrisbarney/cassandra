"""Recompute diagnostics for complete canonical attempts, including failed fits."""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.bayesian_controls import diagnostic_report
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routing', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    routing = json.loads(args.routing.read_text(encoding='utf-8'))
    folders = sorted({Path(p) for group in ('primary', 'sensitivity')
                      for p in routing[group].values()})
    source = Path('src/cyber_threat_modelling/models/bayesian_controls.py')
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        if not folder.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError('Posterior directory escapes project.')
        for attempt_path in sorted(folder.glob('attempt_*.json')):
            attempt = json.loads(attempt_path.read_text(encoding='utf-8'))
            retry = attempt['retry']
            posterior_path = folder / f'posterior_{retry}.nc'
            if attempt.get('early_failure'):
                if posterior_path.exists():
                    raise ValueError('Early-stopped attempt must not have a full posterior.')
                continue
            digest = file_hash(posterior_path)
            identity = {'directory': folder.as_posix(), 'retry': retry,
                        'posterior_sha256': digest, 'attempt_sha256': file_hash(attempt_path),
                        'diagnostic_source_sha256': file_hash(source)}
            output = args.output / f'{digest}.json'
            if output.exists():
                saved = json.loads(output.read_text(encoding='utf-8'))
                if any(saved[k] != v for k, v in identity.items()) or not saved['verified']:
                    raise ValueError('Cached diagnostic review has a different identity.')
                continue
            idata = az.from_netcdf(posterior_path)
            if (idata.posterior.sizes['chain'] != 4 or
                    idata.posterior.sizes['draw'] != attempt['draws']):
                raise ValueError('Saved complete posterior is truncated.')
            actual, _ = diagnostic_report(idata, attempt['max_treedepth'])
            idata.close()
            for key, expected in attempt['diagnostics'].items():
                valid = (not np.isfinite(actual[key]) if expected is None else
                         np.isclose(actual[key], expected, rtol=1e-10, atol=1e-10))
                if not valid:
                    raise ValueError(f'Saved diagnostic differs: {folder} retry {retry} {key}')
            if file_hash(posterior_path) != digest or file_hash(attempt_path) != identity[
                'attempt_sha256'
            ]:
                raise ValueError('Attempt changed during independent review.')
            write_json(output, {**identity, 'verified': True,
                                'reported_diagnostics': attempt['diagnostics'],
                                'script_sha256': file_hash(Path(__file__))})
            print(folder.as_posix(), retry, 'verified', flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
