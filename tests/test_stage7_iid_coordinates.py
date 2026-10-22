import numpy as np
import pytest
from scipy.special import logit

from cyber_threat_modelling.evaluation.synthetic import generate
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID
from cyber_threat_modelling.models.synthetic_state_space import build_model


def test_iid_complete_target_jacobian_and_fused_gradients():
    data, truth, options = generate('dispersion_dense', 7101)
    data = {key: value[:5] if key != 'M' else value for key, value in data.items()}
    data['N'] = data['N'].astype(float)
    data['N'][0, 0] = 0
    data['N'][1, 0] = np.nan
    data['mask_N'][1, 0] = False
    data['effort'] = np.array([-.3, .1, .2, 0., -.1])
    options.update(months=5, prior_id=CURRENT_PRIOR_ID)
    old_model = build_model(data, options, truth)
    new_model = build_model(data, dict(options, residual_parameterization='conditional'), truth)
    old_logp, new_logp = old_model.compile_logp(), new_model.compile_logp()
    gradient = new_model.compile_dlogp()
    fused = new_model.compile_fn([new_model.logp(), new_model.dlogp()],
                                 inputs=new_model.value_vars, mode='NUMBA')
    z = np.linspace(-.3, .3, 5)
    for tau, psi in [(1e-6, .7), (.3, 12.), (.9, 1500.)]:
        old, new = old_model.initial_point(), new_model.initial_point()
        for point in (old, new):
            point['tau_interval__'] = np.array([logit(tau)])
            point['psi_log__'] = np.array([np.log(psi)])
        # Independent diagonal Gaussian posterior algebra in innovation coordinates.
        precision = np.eye(5)
        natural = np.zeros(5)
        for t in range(5):
            if not data['mask_N'][t, 0]:
                continue
            y = data['N'][t, 0]
            variance = 1/(y+.5)+1/psi
            precision[t, t] += tau**2/variance
            natural[t] = tau*(np.log(y+.5)-truth['m'][0]-data['effort'][t])/variance
        covariance = np.linalg.inv(precision)
        chol = np.linalg.cholesky(covariance)
        innovation = covariance @ natural+chol @ z
        new['u_whitened'] = z[:, None]
        old['u_raw'] = innovation[:, None]
        expected_jacobian = np.log(np.diag(chol)).sum()
        assert np.allclose(new_logp(new)-old_logp(old), expected_jacobian, atol=2e-8)
        native = gradient(new)
        compiled_value, compiled_gradient = fused(new)
        assert np.isfinite(native).all()
        assert np.allclose(compiled_value, new_logp(new), atol=2e-8)
        assert np.allclose(compiled_gradient, native, rtol=1e-7, atol=1e-7)
        offset = 0
        for variable in new_model.value_vars:
            name = variable.name
            for index in range(new[name].size):
                step = np.zeros_like(new[name])
                step.flat[index] = 1e-5
                plus, minus = dict(new), dict(new)
                plus[name], minus[name] = new[name]+step, new[name]-step
                finite_difference = (new_logp(plus)-new_logp(minus))/(2e-5)
                assert np.allclose(native[offset+index], finite_difference,
                                   rtol=3e-5, atol=3e-5)
            offset += new[name].size


def test_iid_coordinates_reject_unvalidated_combinations():
    data, truth, options = generate('dispersion_dense', 7101)
    options.update(prior_id=CURRENT_PRIOR_ID, residual_parameterization='conditional')
    for change in [{'factors': 1}, {'regimes': 2}, {'kernel': True},
                   {'residual': 'ar'}, {'conditional_level': False}]:
        with pytest.raises(ValueError, match='isolated noise-separation'):
            build_model(data, dict(options, **change), truth)
