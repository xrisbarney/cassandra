"""Describe complete-chain exploration before any synthetic recovery assessment."""

import json
from contextlib import closing
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'docs/simulations/stage7-reauthorized-20261023'
    audit = json.loads((output/'diagnostics.json').read_text())
    rows = []
    for row in audit['cases']:
        case = row['case']
        if not row['diagnostics_independently_recomputed']:
            rows.append({'case': case, 'review': 'No complete four-chain posterior.'})
            continue
        folder = root/'artifacts/stage7/reauthorized-20261023/fits'/case
        result = row['result']
        assert file_hash(folder/'posterior.nc') == result['artifacts']['posterior.nc']
        summaries = {}
        with closing(az.from_netcdf(folder/'posterior.nc')) as idata:
            posterior = idata.posterior
            arrays = {name: posterior[name].values.reshape(4, 2000, -1)
                      for name in ('m_sd', 'm', 'm_fallback', 'f_diag', 'f_free', 'psi',
                                   'joint_level_whitened')}
            if 'f_phi' in posterior:
                arrays['f_phi'] = posterior.f_phi.values.reshape(4, 2000, -1)
            arrays['factor_time_mean'] = posterior.f_states.values.mean(axis=2)
            for name, x in arrays.items():
                summaries[name] = {
                    'chain_means': x.mean(axis=1).tolist(),
                    'chain_quantiles_05_50_95': np.quantile(x, [.05, .5, .95], axis=1).tolist(),
                }
                if name in ('f_free', 'f_diag'):
                    signs = x > 0
                    summaries[name].update(
                        positive_fractions=signs.mean(axis=1).tolist(),
                        sign_changes=np.count_nonzero(np.diff(signs, axis=1), axis=1).tolist())
            # Include the fixed fit's flagged loading and worst R-hat coordinate.
            selected = [('m_sd', 0), ('m_fallback', 0), ('f_diag', 0),
                        ('factor_time_mean', 0), ('f_free', 8), ('joint_level_whitened', 4)]
            if 'f_phi' in arrays:
                selected.insert(0, ('f_phi', 0))
            fig, axes = plt.subplots(len(selected), 2, figsize=(12, 2.4*len(selected)))
            for axes_row, (name, index) in zip(axes, selected, strict=True):
                for chain in range(4):
                    x = arrays[name][chain, :, index]
                    axes_row[0].plot(x, alpha=.6, linewidth=.5, label=f'Chain {chain}')
                    axes_row[1].hist(x, bins=45, density=True, histtype='step',
                                     label=f'Chain {chain}')
                axes_row[0].set_ylabel(f'{name}[{index}]')
                axes_row[0].set_xlabel('Retained draw')
                axes_row[1].set_xlabel(f'{name}[{index}]')
            axes[0, 0].legend(ncol=4, fontsize=8)
            fig.suptitle(f'{case}: complete chains; descriptive exploration review')
            fig.tight_layout()
            figure = output/(case+'-chains.png')
            if figure.exists():
                raise FileExistsError(figure)
            fig.savefig(figure, dpi=140)
            plt.close(fig)
            correlation_names = ['m_sd', 'm_fallback', 'f_diag', 'factor_time_mean']
            if 'f_phi' in arrays:
                correlation_names.append('f_phi')
            correlation = np.corrcoef(np.stack(
                [arrays[name][:, :, 0].ravel() for name in correlation_names]))
            rows.append({
                'case': case, 'numerical_status': row['status'],
                'posterior_sha256': file_hash(folder/'posterior.nc'),
                'chain_summaries': summaries,
                'correlation_variable_order': correlation_names,
                'pooled_correlation_matrix': correlation.tolist(),
                'figure': figure.relative_to(root).as_posix(),
                'figure_sha256': file_hash(figure),
            })
    path = output/'exploration.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, {
        'cases': rows, 'source_sha256': file_hash(Path(__file__)),
        'diagnostic_audit_sha256': file_hash(output/'diagnostics.json'),
        'interpretation': 'Descriptive checks only; not proof of global exploration. '
                          'No new numerical admission thresholds or fits. '
                          'Earlier divergent callback states are not mode-weight estimates.',
        'visual_review_pending': True, 'real_data_read': False,
    })


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
