"""Explicit prior identity and truth-coordinate conversion for Stage 7 only."""

import hashlib
import json

import numpy as np

PRIOR_ID = 'stationary_bounded_v1'
CURRENT_PRIOR_ID = 'anchored_levels_v2'
CANDIDATES = {
    'primary': (1., 1., 2., 1.),
    'level_tight': (.75, .75, 1.5, 1.),
    'level_wide': (1.5, 1.5, 3., 1.),
    'dynamics_tight': (1., 1., 2., .75),
    'dynamics_wide': (1., 1., 2., 1.5),
}


def verify_activation(root, prior_id=PRIOR_ID):
    filename = ('STATE_SPACE_LEVELS_20261021.json' if prior_id == CURRENT_PRIOR_ID else
                'STATE_SPACE_PRIORS_ACTIVATION_20261021.json')
    path = root / 'docs/amendments' / filename
    record = json.loads(path.read_text())
    if record['status'] != 'active' or record['prior_id'] != prior_id:
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
    if prior_id == CURRENT_PRIOR_ID:
        verify_activation(root, PRIOR_ID)
    else:
        draft = json.loads((root / record['proposal_record']).read_text())
        for entry in draft['files']:
            if digest(root / entry['path']) != entry['sha256']:
                raise ValueError('A sealed prior proposal artifact changed.')
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(path), **record}


def amended(options):
    identity = options.get('prior_id', 'frozen_v1')
    if identity not in ('frozen_v1', PRIOR_ID, CURRENT_PRIOR_ID):
        raise ValueError('Unknown Stage 7 prior identity.')
    return identity != 'frozen_v1'


def level_settings(data, options):
    """Training-prefix anchor; never inspect truth, future targets or zero status."""
    candidate = options.get('prior_candidate', 'primary')
    if candidate not in CANDIDATES:
        raise ValueError('Unknown prior sensitivity candidate.')
    hyper, scale, upper, amplitude = CANDIDATES[candidate]
    counts = np.asarray(data['N'], dtype=float)
    mask = np.asarray(data['mask_N'], dtype=bool)
    months, k = counts.shape
    prefix = min(108, months)
    fallback = options.get('fallback_index')
    if fallback is not None and (not isinstance(fallback, int) or not 0 <= fallback < k):
        raise ValueError('Invalid declared fallback index.')
    known = [j for j in range(k) if j != fallback]
    if not known:
        raise ValueError('At least one known category is required.')
    def mean(indices):
        valid = mask[:prefix, indices]
        selected = counts[:prefix, indices][valid]
        if not len(selected) or not np.isfinite(selected).all() or (selected < 0).any():
            raise ValueError('Anchor requires finite observed training counts.')
        return float(selected.mean())
    return {
        'anchor': float(np.log(mean(known) + .5)), 'training_months': prefix,
        'known_indices': known, 'fallback_index': fallback,
        'fallback_anchor': float(np.log(mean([fallback]) + .5)) if fallback is not None else None,
        'hyper_sd': hyper, 'scale_sd': scale, 'scale_upper': upper,
        'amplitude_multiplier': amplitude, 'candidate': candidate,
    }


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
