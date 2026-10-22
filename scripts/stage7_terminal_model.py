"""Isolated one-factor diagnostic builder; sealed Stage 7 sources stay unchanged."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from cyber_threat_modelling.models.joint_coordinates import joint_count_states
from cyber_threat_modelling.models.synthetic_state_space import loading_matrix


def build_joint(data, options, *, fixed_phi=None, restricted_bound=None):
    if options['factors'] != 1 or options['regimes'] != 1:
        raise ValueError('Terminal diagnostics require one factor and one regime.')
    if options['residual'] != 'none' or options['kernel'] or options['severity_factors']:
        raise ValueError('Terminal diagnostics require the count-only factor model.')
    if fixed_phi is not None and restricted_bound is not None:
        raise ValueError('Fixed and restricted persistence are different diagnostics.')
    t, k = data['N'].shape
    settings = options['level_settings']
    known, fallback = settings['known_indices'], settings['fallback_index']
    center = np.full(k, settings['anchor'])
    if fallback is not None:
        center[fallback] = settings['fallback_anchor']
    with pm.Model() as model:
        sd = pm.TruncatedNormal('m_sd', 0, settings['scale_sd'], lower=0,
                                 upper=settings['scale_upper'], initval=.7)
        covariance = pt.zeros((k, k))
        known_cov = sd**2*np.eye(len(known))+settings['hyper_sd']**2
        covariance = pt.set_subtensor(covariance[np.ix_(known, known)], known_cov)
        if fallback is not None:
            covariance = pt.set_subtensor(covariance[fallback, fallback], 1.)
        psi = pm.LogNormal('psi', np.log(10), 1, shape=k, initval=np.full(k, 10))
        if fixed_phi is not None:
            phi = pt.as_tensor_variable(np.asarray(fixed_phi, dtype=float))
        elif restricted_bound is not None:
            unit = pm.Beta('persistence_unit', 2, 2, shape=1, initval=np.array([.7]))
            phi = pm.Deterministic('f_phi', restricted_bound*(2*unit-1))
        else:
            phi = pm.Uniform('f_phi', -1, 1, shape=1, initval=np.array([.4]))
        loading = loading_matrix('f', k, 1, True, settings['amplitude_multiplier'])
        m, contribution = joint_count_states(t, 1, phi, loading, center, covariance, psi,
                                             data['N'], data['mask_N'], data['effort'],
                                             known, fallback)
        eta = pm.Deterministic('eta_by_regime', (m+contribution)[:, None, :])
        months, cats = np.where(data['mask_N'])
        pm.Potential('observations', pm.logp(pm.NegativeBinomial.dist(
            mu=pt.exp(eta[months, 0, cats]+data['effort'][months]), alpha=psi[cats]),
            data['N'][months, cats]).sum())
    return model
