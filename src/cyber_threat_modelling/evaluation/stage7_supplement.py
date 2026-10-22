"""Exact fixtures and guards for the single-attempt Stage 7 supplement."""

import json
from pathlib import Path

import numpy as np

from ..data.store import write_json
from ..models.stage7_priors import CURRENT_PRIOR_ID, level_settings, verify_activation
from ..provenance import file_hash
from .synthetic import generate

DECLARATION = Path('docs/decisions/STAGE7_REPAIR_SUPPLEMENT_20261022.json')
FIXTURES = Path('docs/fixtures/stage7-supplement')
ACTIVATION = Path('docs/decisions/STAGE7_REPAIR_ACTIVATION_20261022.json')


def declaration(root):
    record = json.loads((root / DECLARATION).read_text(encoding='utf-8'))
    for name, key in [(record['plan'], 'plan_sha256'),
                      ('docs/amendments/STATE_SPACE_LEVELS_20261021.json',
                       'parent_prior_activation_sha256'),
                      ('configs/pilot.toml', 'production_config_sha256_unchanged')]:
        if file_hash(root / name) != record[key]:
            raise ValueError('Supplement declaration dependency changed.')
    if record['stage8_enabled'] or record['held_out_scoring_enabled']:
        raise ValueError('The supplement must remain synthetic and outside Stage 8.')
    verify_activation(root, CURRENT_PRIOR_ID)
    return record


def make_fixture(root, case):
    """Derive the declared intervention; never change canonical input files."""
    record = declaration(root)
    selected = next((c for c in record['cases'] if c['id'] == case), None)
    if selected is None:
        raise ValueError('Undeclared supplement case.')
    scenario = selected['source'].rsplit('-', 1)[0]
    original_plan = json.loads((root / 'docs/decisions/STAGE7_HOSTED_PENDING_20261022.json')
                               .read_text(encoding='utf-8'))
    source = next(c for c in original_plan['cases'] if c['scenario'] == scenario
                  and c['seed'] == 7101)
    arrays = []
    for name in ('data', 'truth'):
        path = root / source['fixture_directory'] / (name + '.npz')
        if file_hash(path) != source[name + '_sha256']:
            raise ValueError('Changed canonical fixture.')
        with np.load(path) as saved:
            arrays.append({k: np.array(saved[k], copy=True) for k in saved.files})
    data, truth = arrays
    reference = {k: np.array(v, copy=True) for k, v in data.items()}
    _, _, options = generate(scenario, 7101)
    options.update(prior_id=CURRENT_PRIOR_ID, prior_candidate='primary',
                   factors=selected['factors'], count_parameterization='conditional',
                   gaussian_solver='block', factor_parameterization='anchored',
                   mass_matrix='diagonal', supplement_case=case)
    if selected['mask_fallback_counts']:
        data['mask_N'][:, 9] = False
        options['level_parameterization'] = 'joint_conditional'
    if selected.get('regenerate_counts'):
        truth['phi'] = truth['phi'][:1].copy()
        truth['loading'] = truth['loading'][:, :1].copy()
        truth['f'] = truth['f'][:, :1].copy()
        truth['eta'] = truth['m'] + truth['f'][1:] @ truth['loading'].T
        truth['mu'] = np.exp(truth['eta'])
        rng = np.random.default_rng(record['matched_count_seed'])
        data['N'] = rng.negative_binomial(truth['psi'],
                                        truth['psi'] / (truth['psi'] + truth['mu']))
        future_rng = np.random.default_rng(record['matched_future_state_seed'])
        state = truth['f'][-1].copy()
        states = []
        for _ in range(12):
            state = truth['phi'] * state + np.sqrt(1 - truth['phi']**2) * future_rng.normal()
            states.append(state.copy())
        truth['future_eta'] = truth['m'] + np.stack(states) @ truth['loading'].T
        truth['future_mu'] = np.exp(truth['future_eta'])
        truth['future_counts'] = rng.negative_binomial(
            truth['psi'], truth['psi'] / (truth['psi'] + truth['future_mu']))
        reference = {k: np.array(v, copy=True) for k, v in data.items()}
    options['level_settings'] = level_settings(reference, options)
    return data, truth, reference, options


def prepare(root):
    record = declaration(root)
    output = root / FIXTURES
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in record['cases']:
        data, truth, reference, options = make_fixture(root, case['id'])
        directory = output / case['id']
        directory.mkdir()
        hashes = {}
        for name, arrays in [('data', data), ('truth', truth), ('level_reference', reference)]:
            path = directory / (name + '.npz')
            np.savez_compressed(path, **arrays)
            hashes[path.relative_to(root).as_posix()] = file_hash(path)
        write_json(directory / 'options.json', options)
        hashes[(directory / 'options.json').relative_to(root).as_posix()] = file_hash(
            directory / 'options.json')
        cases.append({'id': case['id'], 'files_sha256': hashes})
    write_json(output / 'manifest.json', {'declaration_sha256': file_hash(root / DECLARATION),
                                        'cases': cases, 'held_out_scoring_enabled': False})


def load_fixture(root, case):
    declaration(root)
    manifest = json.loads((root / FIXTURES / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['declaration_sha256'] != file_hash(root / DECLARATION):
        raise ValueError('Fixture declaration changed.')
    selected = next((c for c in manifest['cases'] if c['id'] == case), None)
    if selected is None:
        raise ValueError('Undeclared supplement fixture.')
    for name, digest in selected['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError('Changed or escaping supplement fixture.')
    folder = root / FIXTURES / case
    values = []
    for name in ('data', 'truth', 'level_reference'):
        with np.load(folder / (name + '.npz')) as saved:
            values.append(dict(saved))
    return (*values, json.loads((folder / 'options.json').read_text(encoding='utf-8')))


def verify_fit_authority(root, case):
    declaration(root)
    record = json.loads((root / ACTIVATION).read_text(encoding='utf-8'))
    if record['status'] != 'active' or case not in record['cases']:
        raise ValueError('Supplement case has not been activated.')
    for name, digest in record['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError('Activated supplement source or evidence changed.')
    return record
