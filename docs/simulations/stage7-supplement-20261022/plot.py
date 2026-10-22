"""Fixed-case diagnostic plots; no recovery bands for failed posteriors."""
import argparse
import json
from contextlib import closing
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import load_fixture
from cyber_threat_modelling.evaluation.synthetic_recovery import posterior_array
from cyber_threat_modelling.provenance import file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path.cwd()
data, truth, reference, options = load_fixture(root, args.case)
folder = root/'artifacts/stage7/supplement-v1'/args.case
result = json.loads((folder/'result.json').read_text())
for name, digest in result['artifacts'].items():
    assert file_hash(folder/name) == digest
args.output.mkdir(parents=True, exist_ok=False)
colors = plt.get_cmap('tab10').colors[:4]
artifacts = {}

if (folder/'posterior.nc').exists():
    with closing(az.from_netcdf(folder/'posterior.nc')) as idata:
        p = idata.posterior
        rows = [(f'factor {j+1} persistence', p['f_phi'].values[:, :, j])
                for j in range(options['factors'])]
        rows += [('known-category level scale', p['m_sd'].values),
                 ('category 10 log level', p['m'].values[:, :, 9])]
        fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.4*len(rows)), squeeze=False)
        for i, (label, values) in enumerate(rows):
            for chain in range(4):
                axes[i, 0].plot(values[chain], lw=.5, alpha=.65, color=colors[chain],
                                label=f'chain {chain+1}')
                axes[i, 1].hist(values[chain], bins=40, density=True, histtype='step',
                                color=colors[chain], label=f'chain {chain+1}')
            axes[i, 0].set_ylabel(label)
            axes[i, 0].set_xlabel('retained draw')
            axes[i, 1].set_xlabel(label)
            axes[i, 1].set_ylabel('within-chain density')
        axes[0, 0].legend(ncol=4, fontsize=8)
        fig.suptitle(f'{args.case}: {result["status"]}\n'
                     'Chain displays do not establish posterior region weights.')
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            name = 'chains.'+suffix
            fig.savefig(args.output/name, dpi=140)
            artifacts[name] = file_hash(args.output/name)
        plt.close(fig)
        if result['status'] == 'accepted':
            eta = posterior_array(p, 'eta_by_regime')[:, :, 0]
            q = np.quantile(eta, [.05, .5, .95], axis=0)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            months = np.arange(1, 109)
            for axis, category in zip(axes, [0, 1, 9]):
                axis.fill_between(months, q[0, :, category], q[2, :, category],
                                  alpha=.25, label='90% posterior interval')
                axis.plot(months, q[1, :, category], label='posterior median', lw=1)
                axis.plot(months, truth['eta'][:, category], color='black', lw=.8,
                          label='generating log intensity')
                axis.set_ylabel(f'category {category+1}\nlog intensity')
            axes[0].legend(ncol=3, fontsize=8)
            axes[-1].set_xlabel('synthetic training month')
            fig.suptitle(f'{args.case}: numerical pass; exploration review remains separate\n'
                         'Fixed displayed categories 1, 2 and 10; pointwise intervals.')
            fig.tight_layout()
            for suffix in ('png', 'pdf'):
                name = 'recovery.'+suffix
                fig.savefig(args.output/name, dpi=140)
                artifacts[name] = file_hash(args.output/name)
            plt.close(fig)
else:
    fig, axis = plt.subplots(figsize=(9, 3))
    axis.axis('off')
    axis.text(.05, .65, f'{args.case}: retained diagnostic failure', fontsize=13)
    axis.text(.05, .35, json.dumps(result['early_failure']), wrap=True, fontsize=10)
    axis.text(.05, .1, 'No complete posterior; no recovery plot.', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output/'failure.png', dpi=140)
    artifacts['failure.png'] = file_hash(args.output/'failure.png')
    plt.close(fig)

write_json(args.output/'audit.json', {
    'case': args.case, 'result_sha256': file_hash(folder/'result.json'),
    'status': result['status'], 'artifacts': artifacts,
    'script_sha256': file_hash(Path(__file__)),
    'interpretation': 'Four-chain diagnostic plots preserve failed cases. Recovery bands '
                      'are drawn only for numerical passes, not a final exploration decision. '
                      'Reduced cases approximate a two-factor generating truth.'})
