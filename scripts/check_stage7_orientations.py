"""Inspect retained loading orientations without using truth or recovery scores.

This is a warning screen, not proof of global exploration or a mode-weight
estimator. Earlier unmixed chains can also be in low-probability regions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for path in sorted({p for root in args.roots for p in root.rglob('result.json')}):
        result = json.loads(path.read_text())
        if 'attempts' not in result:
            continue
        attempts, warnings, earlier = [], [], {}
        for attempt in result['attempts']:
            file = path.parent/f"posterior_{attempt['retry']}.nc"
            if not file.exists():
                continue
            if file_hash(file) != result['artifacts'][file.name]:
                raise ValueError(f'Changed posterior: {file}')
            variables = {}
            with xr.open_dataset(file, group='posterior') as post:
                if post.sizes['chain'] != 4:
                    continue
                for name in ['f_free', 'h_free', 'a_remaining', 'f_working_free']:
                    if name not in post:
                        continue
                    x = post[name].values
                    indicator = x > 0
                    frequency = indicator.mean(axis=1)
                    variables[name] = dict(
                        chain_means=x.mean(axis=1).tolist(),
                        positive_fractions=frequency.tolist(),
                        sign_changes=np.count_nonzero(np.diff(indicator, axis=1), axis=1).tolist(),
                    )
                    if attempt['accepted']:
                        for j in range(frequency.shape[1]):
                            prior = earlier.get(name, [])
                            common_positive = np.all(frequency[:, j] > .999)
                            common_negative = np.all(frequency[:, j] < .001)
                            opposite = any(
                                (common_positive and np.any(f[:, j] < .001))
                                or (common_negative and np.any(f[:, j] > .999)) for f in prior
                            )
                            if opposite:
                                warnings.append(dict(
                                    variable=name, index=j,
                                    reason='Earlier retained opposite orientation'))
                    earlier.setdefault(name, []).append(frequency)
            attempts.append(dict(retry=attempt['retry'], numerical_pass=attempt['accepted'],
                                 posterior_sha256=file_hash(file), variables=variables))
        cases.append(dict(directory=path.parent.as_posix(), result_sha256=file_hash(path),
                          status=result['status'], attempts=attempts, warnings=warnings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Preserve earlier orientation audits.')
    write_json(args.output, dict(cases=cases, source_sha256=file_hash(Path(__file__)),
                                interpretation=__doc__, real_data_read=False))


if __name__ == '__main__':
    main()
