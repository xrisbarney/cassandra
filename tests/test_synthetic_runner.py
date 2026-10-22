import json
import runpy
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytensor
import pytest

from cyber_threat_modelling.models import synthetic_state_space


@pytest.mark.parametrize('reason', ['divergence', 'depth'])
@pytest.mark.parametrize('paused', [False, True])
def test_failed_retained_draw_saves_checkpoint_and_never_becomes_a_posterior(
    tmp_path, monkeypatch, reason, paused
):
    script = Path(__file__).resolve().parents[1]/'scripts/run_stage7.py'
    runner = runpy.run_path(str(script))
    model = nullcontext()
    model.initial_point = lambda: {'state': np.zeros(1)}
    model.__enter__ = lambda: model
    monkeypatch.setattr(synthetic_state_space, 'build_model', lambda *args: model)
    monkeypatch.setattr(pm, 'NUTS', lambda **kwargs: object())
    monkeypatch.setattr(pytensor.config, 'cxx', 'compiler-detection-only-in-this-mocked-test')

    seeds = []

    def sample(**kwargs):
        seeds.append(kwargs['random_seed'])
        retry = len(seeds) - 1
        expected = np.random.default_rng(7101 + retry * 100)
        for initial in kwargs['initvals']:
            np.testing.assert_array_equal(initial['state'], expected.normal(0, .1, size=(1,)))
        warnings.warn('fixture numerical warning', RuntimeWarning, stacklevel=1)
        # Warmup violations must be ignored. The retained event is irreversible.
        for tuning, index in [(True, 0), (False, kwargs['tune'])]:
            stats = {'diverging': reason == 'divergence',
                     'depth': 15 if reason == 'depth' else 3,
                     'perf_counter_diff': .01, 'process_time_diff': .01}
            kwargs['callback'](None, SimpleNamespace(
                chain=0, draw_idx=index, tuning=tuning, stats=[stats],
                point={'state': np.array([1.])},
            ))
        raise AssertionError('A retained violation should have stopped the attempt.')

    monkeypatch.setattr(pm, 'sample', sample)
    output = tmp_path/'case'
    command = [str(script), '--scenario', 'independent', '--seed', '7101',
               '--output', str(output)]
    if paused:
        command.append('--pause-after-attempt')
    for retry in range(3 if paused else 1):
        monkeypatch.setattr(sys, 'argv', command + (['--resume-attempts'] if retry else []))
        runner['main']()
        if paused and retry < 2:
            assert not (output / 'result.json').exists()
            assert (output / f'checkpoint_{retry}.json').exists()
    assert seeds == [7101, 7102, 7103]
    record = json.loads((output/'result.json').read_text())
    assert record['status'] == 'failed_diagnostic_gate'
    assert len(record['attempts']) == 3
    for retry, attempt in enumerate(record['attempts']):
        assert not attempt['accepted']
        assert attempt['early_failure']['draw_index'] == attempt['warmup']
        assert attempt['diagnostics']['max_rhat'] is None
        assert (output/f'failed_state_{retry}.npz').exists()
        assert 'fixture numerical warning' in attempt['unresolved_warnings']
        context = next(w for w in attempt['warning_contexts']
                       if w['message'] == 'fixture numerical warning')
        assert context['phase'] == 'warmup' and context['chain'] == 0
        assert context['draw_index'] == 0 and context['stack']
    assert not list(output.glob('posterior_*.nc'))
    assert not (output/'future.npz').exists()
    assert not (output/'recovery.json').exists()
