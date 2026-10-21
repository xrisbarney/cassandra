"""Audit gates must reject incomplete schedules and truncated accepted chains."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID
from cyber_threat_modelling.provenance import file_hash

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1]
                           / 'scripts/audit_stage7_campaign.py'))


def fixture(folder, accepted=False):
    options = dict(prior_id=CURRENT_PRIOR_ID, prior_candidate='primary',
                   scenario='independent', seed=7101)
    run = dict(options=options, real_data_read=False, held_out_scoring_enabled=False,
               prior_amendment={'sha256': 'activation'}, source_control={})
    for filename, key in [('data.npz', 'data_sha256'), ('truth.npz', 'truth_sha256'),
                          ('truth_original_coordinates.npz',
                           'truth_original_coordinates_sha256')]:
        (folder/filename).write_bytes(b'fixed fixture')
        run[key] = file_hash(folder/filename)
    (folder/'run.json').write_text(json.dumps(run))
    attempts = []
    for index, (warmup, draws, target, depth) in enumerate(
            [(1000, 1000, .95, 12), (2000, 1000, .99, 15), (2000, 2000, .99, 15)]):
        attempts.append(dict(retry=index, warmup=warmup, draws=draws, chains=4,
                             target_accept=target, max_treedepth=depth,
                             accepted=accepted, diagnostics={'accepted': accepted},
                             unresolved_warnings=[]))
        if accepted:
            break
    return dict(options=options, attempts=attempts, artifacts={},
                status='accepted' if accepted else 'failed_diagnostic_gate')


def check(folder, record):
    (folder/'result.json').write_text(json.dumps(record))
    return AUDIT['checked_run'](folder, 'primary', 'independent', 7101,
                                {'sha256': 'activation'})


def test_only_exhausted_declared_failure_schedule_is_terminal(tmp_path):
    record = fixture(tmp_path)
    assert check(tmp_path, record)['status'] == 'failed_diagnostic_gate'
    record['attempts'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        check(tmp_path, record)


def test_changed_budget_cannot_close_case(tmp_path):
    record = fixture(tmp_path)
    record['attempts'][1]['warmup'] = 500
    with pytest.raises(ValueError, match='sampling budget'):
        check(tmp_path, record)


def test_claimed_acceptance_requires_complete_four_chain_posterior(tmp_path):
    record = fixture(tmp_path, accepted=True)
    xr.Dataset({'m': (('chain', 'draw'), np.zeros((3, 1000)))}).to_netcdf(
        tmp_path/'posterior_0.nc', group='posterior', engine='h5netcdf')
    with pytest.raises(ValueError, match='Incomplete accepted posterior'):
        check(tmp_path, record)


def test_route_cannot_read_outside_project(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        AUDIT['routed_folder']({'primary': {'case': '../elsewhere'}},
                               'primary', 'case', 'unused', tmp_path)
