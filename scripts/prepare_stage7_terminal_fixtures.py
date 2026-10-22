"""Seal two prescribed matched replications without fitting or selecting data."""

from pathlib import Path

import numpy as np

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.evaluation.stage7_supplement import load_fixture
from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import (
    CURRENT_PRIOR_ID,
    level_settings,
    stationary_truth,
)
from cyber_threat_modelling.provenance import file_hash


def make(seed):
    data, original, options = generate('fallback_zero', seed)
    truth = stationary_truth(original, options)
    truth['phi'] = truth['phi'][:1]
    truth['loading'] = truth['loading'][:, :1]
    truth['f'] = truth['f'][:, :1]
    truth['eta'] = truth['m']+truth['f'][1:] @ truth['loading'].T
    truth['mu'] = np.exp(truth['eta'])
    index = seed-7100
    count_seed, state_seed = 78001+index*100, 78002+index*100
    count_rng, state_rng = np.random.default_rng(count_seed), np.random.default_rng(state_seed)
    data['N'] = count_rng.negative_binomial(truth['psi'], truth['psi']/(truth['psi']+truth['mu']))
    state, states = truth['f'][-1].copy(), []
    for _ in range(12):
        state = truth['phi']*state+np.sqrt(1-truth['phi']**2)*state_rng.normal()
        states.append(state.copy())
    truth['future_eta'] = truth['m']+np.stack(states) @ truth['loading'].T
    truth['future_mu'] = np.exp(truth['future_eta'])
    truth['future_counts'] = count_rng.negative_binomial(
        truth['psi'], truth['psi']/(truth['psi']+truth['future_mu']))
    options.update(factors=1, prior_id=CURRENT_PRIOR_ID, prior_candidate='primary',
                   count_parameterization='conditional', gaussian_solver='block',
                   factor_parameterization='anchored', mass_matrix='diagonal')
    options['level_settings'] = level_settings(data, options)
    return data, truth, options, {'count_seed': count_seed, 'future_state_seed': state_seed}


def main():
    root = Path(__file__).resolve().parents[1]
    saved = load_fixture(root, 'matched_one_factor_fallback')
    derived = make(7101)
    for old, new in zip(saved[:2], derived[:2]):
        assert set(old) == set(new)
        for key in old:
            if np.issubdtype(old[key].dtype, np.inexact):
                np.testing.assert_allclose(old[key], new[key], rtol=1e-14, atol=1e-14)
            else:
                np.testing.assert_array_equal(old[key], new[key])
    output = root/'docs/fixtures/stage7-terminal'
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in [7102, 7103]:
        data, truth, options, generators = make(seed)
        folder = output/str(seed)
        folder.mkdir()
        np.savez_compressed(folder/'data.npz', **data)
        np.savez_compressed(folder/'truth.npz', **truth)
        write_json(folder/'options.json', options)
        records.append({'seed': seed, 'generators': generators, 'files_sha256': {
            p.relative_to(root).as_posix(): file_hash(p) for p in folder.iterdir()}})
    write_json(output/'manifest.json', {'cases': records, 'seed7101_reconstruction_passed': True,
               'source_sha256': file_hash(Path(__file__)), 'data_selection_permitted': False})
    print('Both matched replications sealed; seed 7101 reconstruction matches.')


if __name__ == '__main__':
    main()
