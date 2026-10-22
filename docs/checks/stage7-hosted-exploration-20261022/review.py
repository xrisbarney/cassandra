"""Describe retained rank-two chain regions without admitting any posterior."""

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--retry', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
run = json.loads((args.run / 'run.json').read_text(encoding='utf-8'))
attempt_path = args.run / f'attempt_{args.retry}.json'
attempt = json.loads(attempt_path.read_text(encoding='utf-8'))
posterior_path = args.run / f'posterior_{args.retry}.nc'
if run['real_data_read'] or run['held_out_scoring_enabled'] or run['options']['factors'] != 2:
    raise ValueError('This review is restricted to rank-two synthetic fits.')
with xr.open_dataset(posterior_path, group='posterior') as post:
    phi = post['f_phi'].values[:, :, 1].copy()
    mean = post['f_states'].values[:, :, 1:, 1].mean(axis=2)
    level = post['m'].values[:, :, 1].copy()
    scale = post['m_sd'].values.copy()
if phi.shape != (4, attempt['draws']):
    raise ValueError('Incomplete saved chains.')
regions = {}
for name, mask in {
    'lower_persistence': phi < .8,
    'high_positive_mean': (phi > .95) & (mean > .5),
    'high_negative_mean': (phi > .95) & (mean < -.5),
}.items():
    counts = mask.sum(axis=1)
    record = {'fraction_per_chain': mask.mean(axis=1).tolist(),
              'transitions_per_chain': (mask[:, 1:] != mask[:, :-1]).sum(axis=1).tolist()}
    if np.all((counts > 0) & (counts < phi.shape[1])):
        record.update(indicator_rhat=float(az.rhat(mask.astype(float), method='rank')),
                      indicator_bulk_ess=float(az.ess(mask.astype(float), method='bulk')))
    else:
        record['indicator_diagnostics'] = 'At least one chain has a constant indicator.'
    regions[name] = record
args.output.mkdir(parents=True, exist_ok=False)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for chain in range(4):
    for ax, values in zip(axes.flat[:3], [phi, mean, scale], strict=True):
        ax.plot(values[chain], lw=.65, alpha=.7, color=f'C{chain}', label=f'Chain {chain+1}')
    axes[1, 1].scatter(mean[chain, ::3], level[chain, ::3], s=5, alpha=.35,
                       color=f'C{chain}')
for ax, title in zip(axes.flat[:3], ['Second-factor persistence',
                                   'Second-factor mean over observed months',
                                   'Known-category level scale'], strict=True):
    ax.set(title=title, xlabel='Retained draw')
axes[0, 0].set_ylabel('AR coefficient')
axes[0, 1].set_ylabel('Factor mean')
axes[1, 0].set_ylabel('Level standard deviation')
axes[1, 1].set(title='Level / factor-mean dependence', xlabel='Second-factor time mean',
               ylabel='Second category level')
axes[0, 0].legend(ncol=2, fontsize=8)
options = run['options']
fig.suptitle(f"{options['prior_candidate']} / {options['scenario']} / "
             f"{options['seed']} / retry {args.retry}: exploration review", fontsize=12)
for suffix in ('png', 'pdf'):
    fig.savefig(args.output / f'chains.{suffix}', dpi=180)
plt.close(fig)
write_json(args.output / 'review.json', {
    'source_sha256': file_hash(Path(__file__)), 'directory': args.run.as_posix(),
    'options': options, 'posterior_sha256': file_hash(posterior_path),
    'attempt_sha256': file_hash(attempt_path), 'run_sha256': file_hash(args.run / 'run.json'),
    'data_sha256': file_hash(args.run / 'data.npz'), 'retry': args.retry,
    'numerical_diagnostics': attempt['diagnostics'], 'regions': regions,
    'phi2_means_per_chain': phi.mean(axis=1).tolist(),
    'factor2_time_means_per_chain': mean.mean(axis=1).tolist(),
    'interpretation': 'Descriptive chain regions, not posterior region weights or a new '
    'recovery threshold. Markers follow the earlier rank-two review: phi2 < .8, or phi2 > '
    '.95 with time-mean factor2 above .5 or below -.5. Different prior candidates have '
    'different targets. This review itself changes no admission or recovery eligibility.',
    'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
})
