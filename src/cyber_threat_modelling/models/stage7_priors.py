"""Explicit prior identity and truth-coordinate conversion for Stage 7 only."""

import hashlib
import json

import numpy as np

PRIOR_ID = 'stationary_bounded_v1'


def verify_activation(root):
    path = root / 'docs/amendments/STATE_SPACE_PRIORS_ACTIVATION_20261021.json'
    record = json.loads(path.read_text())
    if record['status'] != 'active' or record['prior_id'] != PRIOR_ID:
        raise ValueError('The revised Stage 7 prior is not activated.')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    if digest(root / 'docs/STAGE2_FREEZE.json') != record['parent_freeze_sha256']:
        raise ValueError('The parent freeze changed.')
    parent = json.loads((root / 'docs/STAGE2_FREEZE.json').read_text())
    for entry in parent['files']:
        if digest(root / entry['path']) != entry['sha256']:
            raise ValueError('A frozen design document changed.')
    # Verify the sealed draft, including all of its source/audit dependencies.
    for entry in record['files']:
        if digest(root / entry['path']) != entry['sha256']:
            raise ValueError('The activated prior specification changed.')
    draft = json.loads((root / record['proposal_record']).read_text())
    for entry in draft['files']:
        if digest(root / entry['path']) != entry['sha256']:
            raise ValueError('A sealed prior proposal artifact changed.')
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(path), **record}


def amended(options):
    identity = options.get('prior_id', 'frozen_v1')
    if identity not in ('frozen_v1', PRIOR_ID):
        raise ValueError('Unknown Stage 7 prior identity.')
    return identity == PRIOR_ID


def stationary_truth(truth, options):
    """Preserve observations/latent contributions; translate amplitude coordinates."""
    result = {key: np.array(value, copy=True) for key, value in truth.items()}
    if options['factors']:
        scale = np.sqrt(1 - result['phi']**2)
        result['loading'] /= scale
        result['f'] *= scale
    if options['severity_factors']:
        result['loading_h'] /= np.sqrt(1 - result['phi_h']**2)
    if options['residual'] == 'ar':
        result['tau'] /= np.sqrt(1 - result['phi_u']**2)
    return result
