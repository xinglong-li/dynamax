import blackjax
from collections import OrderedDict
from functools import partial
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from dynamax.abstractions import SSM
from dynamax.cond_moments_gaussian_filter.cmgf import (
    iterated_conditional_moments_gaussian_filter as cmgf_filt,
    iterated_conditional_moments_gaussian_smoother as cmgf_smooth,
    EKFIntegrals)
from dynamax.cond_moments_gaussian_filter.generalized_gaussian_ssm import GGSSMParams
from dynamax.linear_gaussian_ssm.inference import (
    LGSSMParams,
    lgssm_filter,
    lgssm_smoother,
    lgssm_posterior_sample)
from dynamax.parameters import (
    to_unconstrained,
    from_unconstrained,
    log_det_jac_constrain)
from dynamax.utils import (
    pytree_sum,
    pytree_stack,
    ensure_array_has_batch_dim)
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    Poisson as Pois)
from tqdm.auto import trange


class _StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.

    The covariance matrix of the latent dynamics model takes the form:
    R @ Q, where Q is a dense matrix (blockwise diagonal),
    and R is the sparsing matrix. For example,
    for an STS model for a 1-d time series with a local linear component
    and a seasonal component with 4 seasons:
                                        | 1, 0, 0 |
                | v1,   0,  0 |         | 0, 1, 0 |
            Q = |  0,  v2,  0 |,    R = | 0, 0, 1 |
                |  0,   0, v3 |         | 0, 0, 0 |
                                        | 0, 0, 0 |
    """

    def __init__(self,
                 param_props,
                 param_priors,
                 params,
                 trans_mat_getters,
                 trans_cov_getters,
                 obs_mats,
                 cov_select_mats,
                 initial_distributions,
                 reg_func=None):
        self.params = params
        self.param_props = param_props
        self.param_priors = param_priors
        self.trans_mat_getters = trans_mat_getters
        self.trans_cov_getters = trans_cov_getters
        self.component_obs_mats = obs_mats
        self.cov_select_mats = cov_select_mats

        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in initial_distributions.values()])
        self.initial_cov = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in initial_distributions.values()])

        self.obs_mat = jnp.concatenate(list(obs_mats.values()), axis=1)
        self.cov_select_mat = jsp.linalg.block_diag(*cov_select_mats.values())

        self.dim_obs, self.dim_state = self.obs_mat.shape
        self.regression = reg_func

    @property
    def emission_shape(self):
        return (self.dim_obs,)

    def log_prior(self, params):
        """Log prior probability of parameters."""
        lps = tree_map(lambda prior, param: prior.log_prob(param), self.param_priors, params)
        return pytree_sum(lps)

    # Instantiate distributions of the SSM model
    def initial_distribution(self):
        """Distribution of the initial state of the SSM form of the STS model.
        """
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
           multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, covariates=None):
        """Depends on the distribution family of the observation.
        """
        raise NotImplementedError

    @jit
    def sample(self, params, key, num_timesteps, covariates=None):
        """Sample a sequence of latent states and emissions with the given parameters.
        """
        if covariates is not None:
            # When there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[-1], covariates)
            get_trans_mat = partial(self.get_trans_mat, params=params[:-1])
            get_trans_cov = partial(self.get_trans_cov, params=params[:-1])
        else:
            inputs = jnp.zeros((num_timesteps, self.dim_obs))
            get_trans_mat = partial(self.get_trans_mat, params=params)
            get_trans_cov = partial(self.get_trans_cov, params=params)

        def _step(prev_state, args):
            key, input, t = args
            key1, key2 = jr.split(key, 2)
            next_state = get_trans_mat(t) @ prev_state
            next_state = next_state + self.cov_select_mat @ MVN(
                jnp.zeros(self.dim_state), get_trans_cov(t)).sample(seed=key1)
            obs = self.emission_distribution(prev_state, input).sample(seed=key2)
            return next_state, (prev_state, obs)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the remaining emissions and states
        key2s = jr.split(key2, num_timesteps)
        _, (states, time_series) = lax.scan(
            _step, initial_state, (key2s, inputs, jnp.arange(num_timesteps)))
        return states, time_series

    @jit
    def marginal_log_prob(self, params, obs_time_series, covariates=None):
        """Compute log marginal likelihood of observations."""
        if covariates is not None:
            inputs = self.regression(params[-1], covariates)
            ssm_params = self._to_ssm_params(params[:-1])
        else:
            inputs = jnp.zeros(obs_time_series.shape)
            ssm_params = self._to_ssm_params(params)
        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=inputs)
        return filtered_posterior.marginal_loglik

    @jit
    def posterior_sample(self, params, key, obs_time_series, covariates=None):
        if covariates is not None:
            inputs = self.regression(params[-1], covariates)
            ssm_params = self._to_ssm_params(params[:-1])
        else:
            inputs = jnp.zeros(obs_time_series.shape)
            ssm_params = self._to_ssm_params(params)

        key1, key2 = jr.split(key, 2)
        ll, states = self._ssm_posterior_sample(key1, ssm_params, obs_time_series)
        unc_obs_means = states @ self.obs_mat.T + inputs
        obs_means = self._emission_constrainer(unc_obs_means)
        key2s = jr.split(key2, obs_time_series.shape[0])
        obs_sampler = lambda state, input, key:\
            self.emission_distribution(state, input).sample(seed=key)
        obs = vmap(obs_sampler)(states, inputs, key2s)
        return obs_means, obs

    def component_posterior(self, params, obs_time_series, covariates=None):
        """Smoothing by component.
        """
        if covariates is not None:
            inputs = self.regression(params[-1], covariates)
            ssm_params = self._to_ssm_params(params[:-1])
        else:
            inputs = jnp.zeros(obs_time_series.shape)
            ssm_params = self._to_ssm_params(params)

        # Compute the posterior of the joint SSM model
        pos = self._ssm_smoother(ssm_params, obs_time_series, inputs)
        mu_pos = pos.smoothed_means
        var_pos = pos.smoothed_covariances

        # Decompose by component
        component_pos = OrderedDict()
        _loc = 0
        for c, obs_mat in self.component_obs_mats.items():
            c_dim = obs_mat.shape[1]
            c_mean = mu_pos[:, _loc:_loc+c_dim]
            c_cov = var_pos[:, _loc:_loc+c_dim, _loc:_loc+c_dim]
            c_obs_mean = vmap(jnp.matmul, (None, 0))(obs_mat, c_mean)
            c_obs_constrained_mean = self._emission_constrainer(c_obs_mean)
            c_obs_cov = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_cov, obs_mat)
            component_pos[c] = (c_obs_constrained_mean, c_obs_cov)
            _loc += c_dim
        return component_pos

    def fit_hmc(self,
                initial_params,
                param_props,
                key,
                num_samples,
                emissions,
                covariates=None,
                warmup_steps=100,
                num_integration_steps=30,
                verbose=True):
        """Sample parameters of the model using HMC."""
        # Make sure the emissions and covariates have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_covariates = ensure_array_has_batch_dim(covariates, self.covariates_shape)

        initial_unc_params, fixed_params = to_unconstrained(initial_params, param_props)

        # The log likelihood that the HMC samples from
        def unnorm_log_pos(_unc_params):
            params = from_unconstrained(_unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = vmap(partial(self.marginal_log_prob, params))(batch_emissions, batch_covariates)
            lp = log_pri + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptations
        hmc_initial_position, fixed_params = to_unconstrained(self.params, self.param_props)
        warmup = blackjax.window_adaptation(blackjax.nuts, unnorm_log_pos, num_steps=warmup_steps)
        init_key, key = jr.split(key)
        hmc_initial_state, hmc_kernel, _ = warmup.run(init_key, initial_unc_params)

        @jit
        def hmc_step(hmc_state, step_key):
            next_hmc_state, _ = hmc_kernel(step_key, hmc_state)
            params = from_unconstrained(hmc_state.position, fixed_params, param_props)
            return next_hmc_state, params

        # Start sampling
        log_probs = []
        samples = []
        hmc_state = hmc_initial_state
        pbar = trange(num_samples) if verbose else range(num_samples)
        for _ in pbar:
            step_key, key = jr.split(key)
            hmc_state, params = hmc_step(hmc_state, step_key)
            log_probs.append(-hmc_state.potential_energy)
            samples.append(params)

        # Combine the samples into a single pytree
        return pytree_stack(samples), jnp.array(log_probs)

    def forecast(self, key, params, obs_time_series, num_forecast_steps,
                 past_covariates=None, forecast_covariates=None):
        """Forecast the time series"""

        if forecast_covariates is not None:
            past_inputs = self.regression(params[-1], past_covariates)
            forecast_inputs = self.regression(params[-1], forecast_covariates)
            ssm_params = self._to_ssm_params(params[:-1])
            get_trans_mat = partial(self.get_trans_mat, params=params[:-1])
            get_trans_cov = partial(self.get_trans_cov, params=params[:-1])
        else:
            past_inputs = jnp.zeros(obs_time_series.shape)
            forecast_inputs = jnp.zeros((num_forecast_steps, self.dim_obs))
            ssm_params = self._to_ssm_params(params)
            get_trans_mat = partial(self.get_trans_mat, params=params)
            get_trans_cov = partial(self.get_trans_cov, params=params)

        # Filtering the observed time series to initialize the forecast
        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        t0 = obs_time_series.shape[0]
        spars_cov = self.cov_select_mat @ get_trans_cov[t0] @ self.cov_select_mat.T
        initial_mean = get_trans_mat[t0] @ filtered_mean[-1]
        initial_cov = get_trans_mat[t0] @ filtered_cov[-1] @ self.get_trans_mat[t0].T + spars_cov
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)

        def _step(prev_params, args):
            key, forecast_input, t = args
            key1, key2 = jr.split(key)
            prev_mean, prev_cov, prev_state = prev_params

            obs_mean_unc = self.obs_mat @ prev_mean + forecast_input
            obs_mean = self._emission_constrainer(obs_mean_unc)
            obs_cov = self.obs_mat @ prev_cov @ self.obs_mat.T
            obs = self.emission_distribution(prev_state, forecast_input).sample(seed=key2)

            next_mean = self.get_trans_mat[t] @ prev_mean
            next_cov = self.get_trans_mat[t] @ prev_cov @ self.get_trans_mat[t].T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + self.cov_select_mat @ MVN(jnp.zeros(self.dim_state),
                                            get_trans_cov[t]).sample(seed=key1)
            return (next_mean, next_cov, next_state), (obs_mean, obs_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov, initial_state)
        carrys = (keys, forecast_inputs, t0+jnp.arange(num_forecast_steps))
        _, (ts_means, ts_mean_covs, ts) = lax.scan(_step, initial_params, carrys)

        return ts_means, ts_mean_covs, ts

    @jit
    def get_trans_mat(self, params, t):
        trans_mat = []
        for c_name, c_params in params.items():
            trans_getter = self.trans_mat_getters[c_params]
            c_trans_mat = trans_getter(c_params, t)
            trans_mat.append(c_trans_mat)
        return jsp.blockdiag(trans_mat)

    @jit
    def get_trans_cov(self, params, t):
        trans_cov = []
        for c_name, c_params in params.items():
            cov_getter = self.trans_cov_getters[c_params]
            c_trans_cov = cov_getter(c_params, t)
            trans_cov.append(c_trans_cov)
        return jnp.blockdiag(trans_cov)

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        raise NotImplementedError

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        raise NotImplementedError

    def _ssm_smoother(self, params, emissions, inputs):
        """The smoother of the corresponding SSM model"""
        raise NotImplementedError

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        raise NotImplementedError

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space."""
        raise NotImplementedError


#####################################################################
# SSM classes for STS model with specific observation distributions #
#####################################################################


class GaussianSSM(_StructuralTimeSeriesSSM):
    """SSM classes for STS model where the observations follow multivariate normal distributions.
    """
    def __init__(self,
                 param_props,
                 param_priors,
                 params,
                 trans_mat_getters,
                 trans_cov_getters,
                 obs_mats,
                 cov_select_mats,
                 initial_distributions,
                 reg_func=None):

        super().__init__(param_props, param_priors, params, trans_mat_getters, trans_cov_getters,
                         obs_mats, cov_select_mats, initial_distributions, reg_func)

    def emission_distribution(self, state, inputs):
        return MVN(self.obs_mat @ state + inputs, self.params['obs_cov'])

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        ts_means, ts_mean_covs, ts = super().forecast(
            key, observed_time_series, num_forecast_steps, past_inputs, forecast_inputs)
        ts_covs = ts_mean_covs + self.params['obs_cov']
        return ts_means, ts_covs, ts

    @jit
    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        get_trans_mat = partial(self.get_trans_mat, params=params)
        sparse_trans_cov = lambda t:\
            self.cov_select_mat @ self.get_trans_cov(params, t) @ self.cov_select_mat
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_cov,
                           dynamics_matrix=get_trans_mat,
                           dynamics_input_weights=jnp.zeros((self.state_dim, 1)),
                           dynamics_bias=jnp.zeros(self.state_dim),
                           dynamics_covariance=sparse_trans_cov,
                           emission_matrix=self.obs_mat,
                           emission_input_weights=jnp.eye(self.dim_state),
                           emission_bias=jnp.zeros(self.dim_obs),
                           emission_covariance=params['obs_cov'])

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return lgssm_filter(params=params, emissions=emissions, inputs=inputs)

    def _ssm_smoother(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return lgssm_smoother(params=params, emissions=emissions, inputs=inputs)

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        return lgssm_posterior_sample(rng=key,
                                      params=ssm_params,
                                      emissions=observed_time_series,
                                      inputs=inputs)

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space.
           Use identity transformation when the observation distribution is MVN.
        """
        return emission


class PoissonSSM(_StructuralTimeSeriesSSM):
    """SSM classes for STS model where the observations follow Poisson distributions.
    """
    def __init__(self,
                 param_props,
                 param_priors,
                 params,
                 trans_mat_getters,
                 trans_cov_getters,
                 obs_mats,
                 cov_select_mats,
                 initial_distributions,
                 reg_func=None):

        super().__init__(param_props, param_priors, params, trans_mat_getters, trans_cov_getters,
                         obs_mats, cov_select_mats, initial_distributions, reg_func)

    def emission_distribution(self, state, inputs):
        log_rate = self.obs_mat @ state + inputs
        return Pois(rate=self._emission_constrainer(log_rate))

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        ts_means, ts_mean_covs, ts = super().forecast(
            key, observed_time_series, num_forecast_steps, past_inputs, forecast_inputs)
        sampler = lambda r, key: Pois(rate=r).sample(seed=key)
        ts_samples = vmap(sampler)(ts_means, jr.split(key, num_forecast_steps))
        return ts_samples, ts_means, ts

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        # NOTE: Currently the GGSSMParams does not depends on time poit t.
        trans_mat = self.get_trans_mat(params, t=0)
        sparse_trans_cov = self.cov_select_mat @ self.get_trans_cov(params, t=0) @ self.cov_select_mat.T
        return GGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_cov,
                           dynamics_function=lambda z: trans_mat @ z,
                           dynamics_covariance=sparse_trans_cov,
                           emission_mean_function=
                               lambda z: self._emission_constrainer(self.obs_mat @ z),
                           emission_cov_function=
                               lambda z: jnp.diag(self._emission_constrainer(self.obs_mat @ z)),
                           emission_dist=lambda mu, _: Pois(log_rate=jnp.log(mu)))

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return cmgf_filt(
            params=params, inf_params=EKFIntegrals(), emissions=emissions, inputs=inputs, num_iter=2)

    def _ssm_smoother(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return cmgf_smooth(
            params=params, inf_params=EKFIntegrals(), emissions=emissions, inputs=inputs, num_iter=2)

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        # TODO:
        # Implement the real posteriror sample.
        # Currently it simply returns the filtered means.
        print('Currently the posterior_sample for STS model with Poisson likelihood\
               simply returns the filtered means.')
        return self._ssm_filter(ssm_params, observed_time_series, inputs)

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space.
        """
        # Use the exponential function to transform the unconstrained rate
        # to rate of the Poisson distribution
        return jnp.exp(emission)
