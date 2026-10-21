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

from cyber_threat_modelling.evaluation.synthetic_recovery import posterior_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    scenarios = ['factors_sparse', 'regimes', 'dispersion_dense', 'kernel_present']
    for ax, scenario in zip(axes.flat, scenarios, strict=True):
        folder = args.runs / f'{scenario}-7101'
        result = json.loads((folder/'result.json').read_text())
        ax.set_title(scenario.replace('_', ' ') + ' (seed 7101)')
        if result['status'] != 'accepted':
            ax.text(.5, .5, 'No admitted posterior\nDiagnostic gate failed',
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


if __name__ == '__main__':
    main()
