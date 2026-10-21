"""Stage 7 research fixtures using recorded component laws, not a real-data adapter.

Array convention is month x category. Nuisance values conditioned on in a fixture
are explicit in its options. The production Stage 8 interface remains separate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..inference.hmm import hmm_logp
from .ar1_transform import ar1_transform
from .components import add_masked, symbolic_mark_logp
from .stage7_priors import CURRENT_PRIOR_ID, amended, level_settings


def loading_matrix(prefix, categories, rank, revised=False, amplitude=1.):
    sd, bound = amplitude / np.sqrt(rank), 2 * amplitude / np.sqrt(rank)
    diag = (pm.TruncatedNormal(prefix + '_diag', 0, sd, lower=0, upper=bound,
                              shape=rank, initval=np.full(rank, .3)) if revised else
            pm.HalfNormal(prefix + "_diag", 1, shape=rank, initval=np.full(rank, 0.3)))
    rows, cols = np.where(np.tril(np.ones((categories, rank)), -1))
    free = (pm.TruncatedNormal(prefix + '_free', 0, sd, lower=-bound, upper=bound,
                              shape=len(rows), initval=np.zeros(len(rows))) if revised else
            pm.Normal(prefix + "_free", 0, 1, shape=len(rows), initval=np.zeros(len(rows))))
    matrix = pt.zeros((categories, rank))
    matrix = pt.set_subtensor(matrix[np.arange(rank), np.arange(rank)], diag)
    return pt.set_subtensor(matrix[rows, cols], free)


def factors(prefix, categories, rank, months, parameterization='innovations', revised=False,
            multiplier=1.):
    phi = pm.Uniform(prefix + "_phi", -1, 1, shape=rank, initval=np.full(rank, 0.4))
    matrix = None
    if parameterization == 'anchored':
        if not revised:
            raise ValueError('Amplitude coordinates require the recorded revised prior.')
        matrix = loading_matrix(prefix, categories, rank, revised, multiplier)
        amplitude = pt.diag(matrix[:rank])
        scaled = pm.Flat(prefix + '_scaled_states', shape=(months + 1, rank),
                         initval=np.zeros((months + 1, rank)))
        initial = pm.logp(pm.Normal.dist(0, amplitude), scaled[0]).sum()
        transitions = pm.logp(pm.Normal.dist(phi*scaled[:-1],
                                              amplitude*pt.sqrt(1-phi**2)), scaled[1:]).sum()
        pm.Potential(prefix + '_state_prior', initial + transitions)
        states = pm.Deterministic(prefix + '_states', scaled / amplitude)
    elif parameterization == 'centered':
        # The Flat base measure plus this normalized AR density is exactly the
        # same stationary Gaussian prior. Its density remains outside the HMM.
        states = pm.Flat(prefix + '_states', shape=(months + 1, rank),
                         initval=np.zeros((months + 1, rank)))
        initial_sd = 1 if revised else 1 / pt.sqrt(1 - phi**2)
        innovation_sd = pt.sqrt(1 - phi**2) if revised else 1
        initial = pm.logp(pm.Normal.dist(0, initial_sd), states[0]).sum()
        transitions = pm.logp(pm.Normal.dist(phi * states[:-1], innovation_sd), states[1:]).sum()
        pm.Potential(prefix + '_state_prior', initial + transitions)
    elif parameterization == 'innovations':
        innovations = pm.Normal(prefix + "_innovations", 0, 1, shape=(rank, months + 1))
        transformed = ar1_transform(innovations, phi).T
        states = pm.Deterministic(prefix + "_states",
                                  transformed * pt.sqrt(1 - phi**2) if revised else transformed)
    else:
        raise ValueError('Unknown factor coordinates.')
    if matrix is None:
        matrix = loading_matrix(prefix, categories, rank, revised, multiplier)
    contribution = pm.Deterministic(prefix + "_contribution", states[1:] @ matrix.T)
    return contribution


def hierarchical_level(prefix, categories, center, hyper_sd, scale_sd, init, revised=False):
    # Integrate the Gaussian group mean exactly. The marginal covariance is
    # s^2 I + hyper_sd^2 11'; this preserves the frozen hierarchical prior while
    # avoiding a likelihood ridge between a group mean and standardized levels.
    sd = (pm.TruncatedNormal(prefix + '_sd', 0, scale_sd, lower=0,
                             upper=3 if prefix == 'm' else 2, initval=1) if revised else
          pm.HalfNormal(prefix + "_sd", scale_sd, initval=1))
    covariance = sd**2 * np.eye(categories) + hyper_sd**2 * np.ones((categories, categories))
    return pm.MvNormal(prefix, mu=np.full(categories, center), cov=covariance, initval=init)


def build_model(data, options, conditional=None):
    revised = amended(options)
    settings = (level_settings(data, options)
                if options.get('prior_id') == CURRENT_PRIOR_ID else None)
    multiplier = settings['amplitude_multiplier'] if settings else 1.
    t, k = data["N"].shape
    r, rank = options["regimes"], options["factors"]
    with pm.Model() as model:
        if options["conditional_level"]:
            m = pt.as_tensor_variable(np.asarray(conditional["m"]))
        else:
            observed_sum = np.where(data['mask_N'], data['N'], 0).sum(axis=0)
            observed_count = data['mask_N'].sum(axis=0)
            initial_rate = np.divide(observed_sum, observed_count,
                                     out=np.zeros(k), where=observed_count > 0)
            if settings:
                known = settings['known_indices']
                sd = pm.TruncatedNormal('m_sd', 0, settings['scale_sd'], lower=0,
                                         upper=settings['scale_upper'], initval=.7)
                size = len(known)
                cov = sd**2*np.eye(size) + settings['hyper_sd']**2*np.ones((size, size))
                levels = pm.MvNormal('m_known', mu=np.full(size, settings['anchor']),
                                     cov=cov, initval=np.log(initial_rate[known]+.5))
                m = pt.set_subtensor(pt.zeros(k)[known], levels)
                if settings['fallback_index'] is not None:
                    fallback = pm.Normal('m_fallback', settings['fallback_anchor'], 1,
                                          initval=settings['fallback_anchor'])
                    m = pt.set_subtensor(m[settings['fallback_index']], fallback)
                m = pm.Deterministic('m', m)
            else:
                m = hierarchical_level("m", k, 0, 3, 2, np.log(initial_rate + 0.5), revised)
        psi = pm.LogNormal("psi", np.log(10), 1, shape=k, initval=np.full(k, 10))
        base = pt.broadcast_to(m, (t, k))
        if options.get('count_parameterization') == 'conditional':
            if not revised or not rank or r != 1 or options['residual'] != 'none' \
                    or options['kernel']:
                raise ValueError('Conditional count coordinates require the factor-only R=1 model.')
            from .gaussian_coordinates import count_states

            phi_f = pm.Uniform('f_phi', -1, 1, shape=rank, initval=np.full(rank, .4))
            matrix_f = loading_matrix('f', k, rank, revised, multiplier)
            base = base + count_states(t, rank, phi_f, matrix_f, m, psi,
                                       data['N'], data['mask_N'], data['effort'])
        elif rank:
            base = base + factors("f", k, rank, t,
                                  options.get('factor_parameterization', 'innovations'), revised,
                                  multiplier)
        if options["residual"] != "none":
            tau = (pm.TruncatedNormal('tau', 0, multiplier, lower=0, upper=multiplier, shape=k,
                                      initval=np.full(k, .3))
                   if revised else
                   pm.Deterministic("tau", 0.01 + pm.HalfNormal("tau_excess", 1, shape=k)))
            if options["residual"] == "iid":
                u = pm.Normal("u_raw", 0, 1, shape=(t, k)) * tau
            else:
                phi_u = pm.Uniform("phi_u", -1, 1, shape=k, initval=np.full(k, 0.4))
                if options.get('factor_parameterization') == 'anchored' and revised:
                    states_u = pm.Flat('u_states', shape=(t+1, k), initval=np.zeros((t+1, k)))
                    pm.Potential('u_state_prior',
                                 pm.logp(pm.Normal.dist(0, tau), states_u[0]).sum()
                                 + pm.logp(pm.Normal.dist(phi_u*states_u[:-1],
                                                         tau*pt.sqrt(1-phi_u**2)),
                                           states_u[1:]).sum())
                    u = states_u[1:]
                else:
                    innovations = pm.Normal("u_innovations", 0, 1, shape=(k, t + 1))
                    u = ar1_transform(innovations, phi_u).T[1:] * tau
                    if revised:
                        u = u * pt.sqrt(1 - phi_u**2)
            base = base + pm.Deterministic("u", u)
        if options["kernel"]:
            width = pm.Truncated("width", pm.LogNormal.dist(np.log(12), 0.5), lower=0, upper=18)
            raw = pm.math.sigmoid((width - np.abs(np.arange(61) - 30)) / 2)
            weights = raw / pt.sqrt(pt.sum(raw**2))
            xi = pm.Normal("xi", 0, 1, shape=t + 60)
            indices = np.arange(t)[:, None] + 60 - np.arange(61)[None, :]
            g = pm.Deterministic("g", pt.sum(xi[indices] * weights, axis=1))
            scale = pm.HalfNormal("a_scale", 0.5*multiplier)
            if revised:
                first = pm.TruncatedNormal('a_first', 0, scale, lower=0,
                                            upper=multiplier, initval=.2)
                remaining = pm.TruncatedNormal('a_remaining', 0, scale,
                                                lower=-multiplier, upper=multiplier,
                                                shape=k-1, initval=np.zeros(k-1))
                a = pm.Deterministic('a', pt.concatenate([first[None], remaining]))
            else:
                first = pm.HalfNormal("a_first_raw", 1)
                remaining = pm.Normal("a_remaining_raw", 0, 1, shape=k - 1)
                a = pm.Deterministic("a", scale * pt.concatenate([first[None], remaining]))
            pm.Deterministic("a_norm", pt.sqrt(pt.sum(a**2)))
            base = base + g[:, None] * a
        if r > 1:
            increments = pm.HalfNormal("regime_increments", 1, shape=r - 1)
            shifts = pt.concatenate([pt.zeros(1), pt.cumsum(increments)])
            transition = pm.Dirichlet("transition", 1 + 9 * np.eye(r), shape=(r, r))
        else:
            shifts, transition = pt.zeros(1), pt.ones((1, 1))
        eta = pm.Deterministic("eta_by_regime", base[:, None, :] + shifts[None, :, None])
        emissions = pt.zeros((t, r))
        months, cats = np.where(data["mask_N"])
        nterm = pm.logp(
            pm.NegativeBinomial.dist(
                mu=pt.exp(eta[months, :, cats] + data["effort"][months, None]),
                alpha=psi[cats][:, None],
            ), data["N"][months, cats, None],
        )
        emissions = add_masked(emissions, nterm, months)
        if options["severity_factors"]:
            n = hierarchical_level("n", k, 1, 1, 1, np.ones(k), revised)
            if options.get('severity_parameterization') == 'conditional':
                if not revised:
                    raise ValueError('Conditional severity coordinates require stationary factors.')
                from .gaussian_coordinates import severity_states

                h_rank = options['severity_factors']
                kappa = pm.HalfNormal('kappa', .5, shape=k, initval=np.full(k, .3))
                phi_h = pm.Uniform('h_phi', -1, 1, shape=h_rank, initval=np.full(h_rank, .4))
                matrix_h = loading_matrix('h', k, h_rank, revised, multiplier)
                loc = n + severity_states(t, h_rank, phi_h, matrix_h, n, kappa,
                                           data['B'], data['mask_B'])
            else:
                loc = n + factors("h", k, options["severity_factors"], t,
                                  options.get('factor_parameterization', 'innovations'), revised,
                                  multiplier)
                kappa = pm.HalfNormal("kappa", 0.5, shape=k, initval=np.full(k, 0.3))
            # This fixture has R=1; joint severity/regime validation belongs to Stage 8.
            loc = pm.Deterministic("severity_location", loc)
            months, cats = np.where(data["mask_B"])
            term = symbolic_mark_logp(
                data["B"][months, cats, None], loc[months, cats][:, None],
                kappa[cats][:, None], upper=10, epsilon=0.005,
            )
            emissions = add_masked(emissions, term, months)
        if options["known_auxiliary_coefficients"]:
            months, cats = np.where(data["mask_E"])
            emissions = add_masked(emissions, symbolic_mark_logp(
                data["E"][months, cats, None], -3 + 0.4 * eta[months, :, cats], 0.5,
            ), months)
            months, cats = np.where(data["mask_V"])
            emissions = add_masked(emissions, pm.logp(pm.NegativeBinomial.dist(
                mu=pt.exp(-3 + 0.5 * eta[months, :, cats]), alpha=2,
            ), data["V"][months, cats, None]), months)
            months, sectors = np.where(data["mask_D"])
            rate = 0.02 * pt.sum(
                pt.exp(eta[months]) * data["M"][sectors, None, :], axis=2
            ) + 0.1
            emissions = add_masked(emissions, pm.logp(
                pm.Poisson.dist(rate), data["D"][months, sectors, None]
            ), months)
        # R=1 shortcut is algebraically identical and avoids a redundant Op.
        pm.Potential("observations", pt.sum(emissions) if r == 1 else hmm_logp(
            emissions, pt.log(transition)
        ))
        model.synthetic_emissions = emissions
        model.synthetic_transition = transition
    return model
