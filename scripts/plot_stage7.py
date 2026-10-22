"""Plot the predeclared first synthetic seed; failed fits are displayed as failures."""

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.synthetic_recovery import posterior_array
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--audit', type=Path,
                        help='Use the routed primary summary instead of original folders.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    scenarios = ['factors_sparse', 'regimes', 'dispersion_dense', 'kernel_present']
    routed = ({(r['scenario'], r['seed']): r
               for r in json.loads(args.audit.read_text())['runs']} if args.audit else {})
    plotted = []
    for ax, scenario in zip(axes.flat, scenarios, strict=True):
        record = routed[scenario, 7101] if args.audit else None
        folder = Path(record['directory']) if record else args.runs / f'{scenario}-7101'
        if not folder.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError(f'Run directory escapes project: {folder}')
        if record and file_hash(folder/'result.json') != record['result_sha256']:
            raise ValueError(f'Changed routed result: {folder}')
        result = json.loads((folder/'result.json').read_text())
        if result['options']['scenario'] != scenario or result['options']['seed'] != 7101:
            raise ValueError(f'Wrong plotted case: {folder}')
        for relative, digest in result['artifacts'].items():
            if file_hash(folder/relative) != digest:
                raise ValueError(f'Changed plotted artifact: {folder/relative}')
        plotted.append({'scenario': scenario, 'seed': 7101, 'directory': folder.as_posix(),
                        'status': result['status'],
                        'recovery_eligible': (record.get('recovery_eligible', True)
                                              if record else result['status'] == 'accepted'),
                        'result_sha256': file_hash(folder/'result.json')})
        ax.set_title(scenario.replace('_', ' ') + ' (seed 7101)')
        if result['status'] != 'accepted':
            ax.text(.5, .5, 'No admitted posterior\nDiagnostic gate failed',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if record and not record.get('recovery_eligible', True):
            ax.text(.5, .5, 'Numerical gates passed\nKnown posterior modes unresolved',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        post = az.from_netcdf(folder/f"posterior_{result['attempts'][-1]['retry']}.nc").posterior
        with np.load(folder/'truth.npz') as loaded:
            truth = dict(loaded)
        if scenario == 'factors_sparse':
            eta = posterior_array(post, 'eta_by_regime')[:, :, 0, 0]
            lo, median, hi = np.quantile(eta, [.05, .5, .95], axis=0)
            months = np.arange(len(median))
            ax.fill_between(months, lo, hi, alpha=.25, label='90% interval')
            ax.plot(months, median, label='Posterior median')
            ax.plot(months, truth['eta'][:, 0], color='black', lw=1, label='Truth')
            ax.set(xlabel='Synthetic month', ylabel='Log intensity, dense category')
        elif scenario == 'regimes':
            record = json.loads((folder/'recovery.json').read_text())['regimes']
            ax.plot(np.asarray(record['smoothed_probabilities'])[:, 1], label='Smoothed P(z=1)')
            ax.plot(truth['regime'], color='black', linestyle='--', label='True regime')
            ax.set(xlabel='Synthetic month', ylabel='Retrospective probability')
        elif scenario == 'dispersion_dense':
            tau = posterior_array(post, 'tau')[:, 0]
            psi = posterior_array(post, 'psi')[:, 0]
            ax.scatter(tau[::10], psi[::10], s=5, alpha=.3, label='Posterior draws')
            ax.scatter(truth['tau'], truth['psi'], marker='x', s=80, color='black', label='Truth')
            ax.set(xlabel='Latent noise SD', ylabel='NB dispersion')
        else:
            width = posterior_array(post, 'width')
            ax.hist(width, bins=30, density=True, alpha=.5, label='Posterior width')
            grid = np.linspace(.01, 18, 300)
            prior = lognorm.pdf(grid, .5, scale=12) / lognorm.cdf(18, .5, scale=12)
            ax.plot(grid, prior, color='gray', linestyle='--', label='Frozen prior')
            ax.axvline(truth['width'], color='black', label='True width')
            ax.set(xlabel='Kernel half-width (months)', ylabel='Density')
        ax.legend(fontsize=8)
    fig.suptitle('Controlled simulation checks; not held-out real-data validation', fontsize=12)
    fig.savefig(args.output/'synthetic_recovery.png', dpi=180)
    fig.savefig(args.output/'synthetic_recovery.pdf')
    plt.close(fig)
    write_json(args.output/'audit.json', {
        'cases': plotted, 'source_sha256': file_hash(Path(__file__)),
        'summary_audit_sha256': file_hash(args.audit) if args.audit else None,
        'artifacts': {p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()},
        'selection': 'Predeclared seed 7101; failed posteriors are shown as failures.',
    })


if __name__ == '__main__':
    main()
