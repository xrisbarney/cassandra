"""Checkpoint identity and interruption retention, without a posterior-admission claim."""

import importlib.util
import json
from contextlib import closing
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytest


def checkpoint_function():
    source = Path(__file__).resolve().parents[1]/'scripts/stage7_chain_checkpoint.py'
    spec = importlib.util.spec_from_file_location('checkpoint_target', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.save_complete_chain


def run(callback=None):
    with pm.Model() as model:
        x = pm.Normal('x')
        pm.Deterministic('exp_x', pm.math.exp(x))
        pm.Normal('y', x, 1, observed=[.3, -.2])
        step = pm.NUTS(target_accept=.9)
        return pm.sample(draws=24, tune=24, chains=2, cores=1, step=step, random_seed=7103,
                         initvals=[{'x': np.array(.1)}, {'x': np.array(-.1)}],
                         progressbar=False, compute_convergence_checks=False,
                         callback=(None if callback is None else
                                   lambda trace, draw: callback(trace, draw, model)))


def test_checkpoints_preserve_exact_draws_stats_and_chain_labels(tmp_path):
    save = checkpoint_function()
    complete = []

    def callback(trace, draw, model):
        if draw.draw_idx == 47:
            complete.append(save(trace, model, tmp_path, draw.chain, 24, 24))

    baseline, actual = run(), run(callback)
    try:
        for name in baseline.posterior:
            np.testing.assert_array_equal(baseline.posterior[name], actual.posterior[name])
        for name in baseline.sample_stats:
            if name not in ('perf_counter_diff', 'perf_counter_start', 'process_time_diff'):
                np.testing.assert_array_equal(baseline.sample_stats[name],
                                               actual.sample_stats[name])
        assert len(complete) == 2
        for chain in (0, 1):
            with closing(az.from_netcdf(tmp_path/f'chain_{chain}.nc')) as saved:
                assert saved.posterior.sizes['draw'] == 24
                np.testing.assert_array_equal(saved.posterior.chain, [chain])
                for group in ('posterior', 'sample_stats'):
                    for name in getattr(actual, group):
                        np.testing.assert_array_equal(getattr(saved, group)[name],
                            getattr(actual, group)[name].sel(chain=[chain]))
    finally:
        baseline.close()
        actual.close()


def test_completed_chain_survives_later_interruption(tmp_path):
    save = checkpoint_function()

    def callback(trace, draw, model):
        if draw.chain == 1 and draw.draw_idx == 3:
            raise RuntimeError('simulated later interruption')
        if draw.draw_idx == 47:
            save(trace, model, tmp_path, draw.chain, 24, 24)

    with pytest.raises(RuntimeError, match='simulated later'):
        run(callback)
    assert json.loads((tmp_path/'chain_0.json').read_text())['retained_draws'] == 24
    with closing(az.from_netcdf(tmp_path/'chain_0.nc')) as saved:
        assert saved.posterior.sizes['draw'] == 24
    assert not (tmp_path/'chain_1.nc').exists()


def test_existing_or_incomplete_trace_is_refused(tmp_path):
    save = checkpoint_function()
    (tmp_path/'chain_0.json').write_text('{}')
    with pytest.raises(FileExistsError):
        save(None, None, tmp_path, 0, 24, 24)
    class Partial:
        chain = 1

        def __len__(self):
            return 30
    with pytest.raises(ValueError, match='exact complete'):
        save(Partial(), None, tmp_path, 1, 24, 24)
