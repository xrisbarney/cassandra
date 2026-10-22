"""A working sign chart preserves the complete target and canonical forecasts."""

import numpy as np
import pytest
from scipy.special import logit

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID
from cyber_threat_modelling.models.synthetic_state_space import build_model


def fixture():
    data, truth, options = generate('effort_proxy', 7102)
    data = {key: value[:4] if key != 'M' else value for key, value in data.items()}
    options.update(months=4, prior_id=CURRENT_PRIOR_ID, factor_parameterization='anchored',
                   count_parameterization='conditional', gaussian_solver='block')
    return data, truth, options


def test_working_sign_chart_density_gradients_and_forecast_reconstruction():
    data, truth, options = fixture()
    old_model = build_model(data, options, truth)
    model = build_model(data, dict(options, working_sign_anchor=3), truth)
    old, point = old_model.initial_point(), model.initial_point()
    native_logp, native_grad = model.compile_logp(), model.compile_dlogp()
    target = model.compile_logp(jacobian=False)
    old_target = old_model.compile_logp(jacobian=False)
    fused = model.compile_fn([model.logp(), model.dlogp()], inputs=model.value_vars, mode='NUMBA')
    outputs = model.replace_rvs_by_values([model[name] for name in
        ['f_diag', 'f_free', 'f_states', 'f_working_states', 'f_contribution']])
    reconstruction = model.compile_fn(outputs, inputs=model.value_vars, on_unused_input='ignore')
    old_states = old_model.compile_fn(old_model.replace_rvs_by_values([old_model['f_states']]),
                                      inputs=old_model.value_vars, on_unused_input='ignore')
    for first in [.12, -.12, 0.]:
        working = np.array([first, .6, .9, 1.2])
        point['f_working_positive_interval__'] = logit(working[3:]/2)
        point['f_working_free_interval__'] = logit((working[:3]+2)/4)
        point['f_whitened_states'] = np.linspace(-.2, .3, 5)[:, None]
        sign = 1 if first >= 0 else -1
        if first:
            old['f_diag_interval__'] = logit(np.abs(working[:1])/2)
            old['f_free_interval__'] = logit((sign*working[1:]+2)/4)
            old['f_whitened_states'] = sign*point['f_whitened_states']
            assert np.allclose(target(point), old_target(old), atol=1e-8)
        diagonal, free, states, working_states, contribution = reconstruction(point)
        canonical = np.r_[diagonal, free][:, None]
        assert np.allclose(canonical[:, 0], sign*working)
        assert np.allclose(states, sign*working_states)
        assert np.allclose(contribution, states[1:] @ canonical.T)
        if first:
            assert np.allclose(states, old_states(old)[0])
        # Conditional future means and innovation covariance do not depend
        # on the chart, including at the zero crossing of the working row 0.
        assert np.allclose(.4*states[-1] @ canonical.T,
                           .4*working_states[-1] @ working[None, :])
        assert np.allclose((1-.4**2)*canonical @ canonical.T,
                           (1-.4**2)*working[:, None] @ working[None, :])
        grad = native_grad(point)
        compiled = fused(point)
        assert np.isfinite(grad).all()
        assert np.allclose(compiled[0], native_logp(point), atol=1e-8)
        assert np.allclose(compiled[1], grad, rtol=1e-7, atol=1e-7)
        offset = 0
        for variable in model.value_vars:
            name = variable.name
            if name in ['f_working_free_interval__', 'f_working_positive_interval__',
                        'f_phi_interval__', 'psi_log__']:
                delta = np.zeros_like(point[name])
                delta.flat[0] = 1e-5
                plus, minus = dict(point), dict(point)
                plus[name], minus[name] = point[name]+delta, point[name]-delta
                fd = (native_logp(plus)-native_logp(minus))/2e-5
                assert np.allclose(fd, grad[offset], rtol=2e-5, atol=2e-5)
            offset += point[name].size


@pytest.mark.parametrize('change', [dict(factors=2), dict(working_sign_anchor=0),
                                    dict(count_parameterization='ordinary'), dict(kernel=True)])
def test_sign_chart_rejects_unsupported_model(change):
    data, truth, options = fixture()
    options.update(working_sign_anchor=3)
    options.update(change)
    with pytest.raises(ValueError, match='Working sign anchor'):
        build_model(data, options, truth)
