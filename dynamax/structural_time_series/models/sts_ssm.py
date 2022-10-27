import blackjax
from collections import OrderedDict
from functools import partial
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jaxopt import LBFGS
from dynamax.abstractions import SSM
from dynamax.cond_moments_gaussian_filter.containers import EKFParams
from dynamax.cond_moments_gaussian_filter.inference import (
    iterated_conditional_moments_gaussian_filter as cmgf_filt,
    iterated_conditional_moments_gaussian_smoother as cmgf_smooth)
from dynamax.linear_gaussian_ssm.inference import (
    LGSSMParams,
    lgssm_filter,
    lgssm_smoother,
    lgssm_posterior_sample)
from dynamax.parameters import (
    to_unconstrained,
    from_unconstrained,
    log_det_jac_constrain,
    flatten,
    unflatten,
    ParameterProperties)
from dynamax.utils import PSDToRealBijector, pytree_sum
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    MultivariateNormalDiag as MVNDiag,
    Poisson)
from tqdm.auto import trange
from dynamax.utils import pytree_stack, ensure_array_has_batch_dim


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
                 params,
                 param_props,
                 priors,
                 trans_mat_getters,
                 obs_mat_getters,
                 trans_cov_getters,
                 initial_distributions,
                 cov_select_mat,
                 regression_comp):
        self.params = params
        self.param_props = param_props
        self.param_priors = priors

        self.component_init_dists = initial_distributions
        self.cov_select_mat = cov_select_mat

        self.trans_mat_getters = OrderedDict()
        self.obs_mat_getters = OrderedDict()
        self.trans_cov_getters = OrderedDict()

        self.dim_obs = None
        self.regression = None

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def log_prior(self, params):
        lps = tree_map(lambda prior, param: prior.log_prob(param), self.param_priors, params)
        return pytree_sum(lps)

    # Instantiate distributions of the SSM model
    def initial_distribution(self):
        """Distribution of the initial state of the SSM model.
        Not implement because some component has 0 covariances.
        """
        return self.initial_state_distribution

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

        Since the regression is contained in the regression component,
        so there is no covariates term under the STS framework.
        """
        if covariates is not None:
            inputs = self.regression(params[-1], covariates)
            get_trans_mat = partial(self.get_trans_mat, params=params[:-1])
            get_trans_cov = partial(self.get_trans_cov, params=params[:-1])
        else:
            inputs = jnp.zeros((num_timesteps, self.dim_obs))
            get_trans_mat = partial(self.get_trans_mat, params=params)
            get_trans_cov = partial(self.get_trans_cov, params=params)
        cov_select_mat = self.cov_select_mat
        dim_comp = get_trans_cov(0).shape[-1]

        def _step(prev_state, args):
            key, input, t = args
            key1, key2 = jr.split(key, 2)
            next_state = get_trans_mat(t) @ prev_state
            next_state = next_state + cov_select_mat @ MVN(jnp.zeros(dim_comp),
                                                           get_trans_cov(t)).sample(seed=key1)
            emission = self.emission_distribution(prev_state, input).sample(seed=key2)
            return next_state, (prev_state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution().sample(seed=key1)
        initial_emission = self.emission_distribution(initial_state, inputs[0]).sample(seed=key2)

        # Sample the remaining emissions and states
        key2s = jr.split(key2, num_timesteps)
        _, (states, emissions) = lax.scan(_step,
                                          initial_state,
                                          (key2s, inputs, jnp.arange(num_timesteps)))
        return states, emissions

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
        num_timesteps, dim_obs = obs_time_series.shape

        ssm_params = self._to_ssm_params(params)

        obs_mats = vmap(self.get_obs_mat, (None, 0))(params, jnp.arange(num_timesteps))

        ll, states = self._ssm_posterior_sample(key1, ssm_params, obs_time_series)
        unc_obs_means = vmap(jnp.matmul)(obs_mats, states) + inputs
        obs_means = self._emission_constrainer(unc_obs_means)
        key2s = jr.split(key2, num_timesteps)
        emission_sampler = lambda state, input, key:\
            self.emission_distribution(state, input).sample(seed=key)
        obs = vmap(emission_sampler)(states, inputs, key2s)
        return obs_means, obs

    def component_posterior(self, params, obs_time_series, covariates=None):
        """Smoothing by component
        """
        num_timesteps, dim_obs = obs_time_series.shape
        # Compute the posterior of the joint SSM model
        if covariates is not None:
            inputs = self.regression(params[-1], covariates)
            ssm_params = self._to_ssm_params(params[:-1])
        else:
            inputs = jnp.zeros(obs_time_series.shape)
            ssm_params = self._to_ssm_params(params)
        component_pos = OrderedDict()
        pos = self._ssm_smoother(ssm_params, obs_time_series, inputs)
        mu_pos = pos.smoothed_means
        var_pos = pos.smoothed_covariances
        obs_mats = vmap(self.get_obs_mat, (None, 0))(params, jnp.arange(num_timesteps))

        # Decompose by component
        _loc = 0
        for c, get_obs_mat in self.obs_mat_getters.items():
            obs_mats = vmap(get_obs_mat, (None, 0))(params[c], jnp.arange(num_timesteps))
            c_dim = obs_mats.shape[-1]
            c_mean = mu_pos[:, _loc:_loc+c_dim]
            c_cov = var_pos[:, _loc:_loc+c_dim, _loc:_loc+c_dim]
            c_obs_mean = vmap(jnp.matmul)(obs_mats, c_mean)
            c_obs_constrained_mean = self._emission_constrainer(c_obs_mean)
            c_obs_cov = vmap(lambda s, m: m @ s @ m.T)(c_cov, obs_mats)
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
        else:
            past_inputs = jnp.zeros(obs_time_series.shape)
            forecast_inputs = jnp.zeros((num_forecast_steps, self.dim_obs))
            ssm_params = self._to_ssm_params(params)

        # Filtering the observed time series to initialize the forecast
        filtered_posterior = self._ssm_filter(params=ssm_params,
                                              emissions=obs_time_series,
                                              inputs=past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        initial_mean = self.dynamics_matrix @ filtered_mean[-1]
        initial_cov = self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T + spars_cov
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)

        def _step(prev_params, args):
            key, forecast_input = args
            key1, key2 = jr.split(key)
            prev_mean, prev_cov, prev_state = prev_params

            marginal_mean = self.emission_matrix @ prev_mean + weights @ forecast_input
            marginal_mean = self._emission_constrainer(marginal_mean)
            emission_mean_cov = self.emission_matrix @ prev_cov @ self.emission_matrix.T
            obs = self.emission_distribution(prev_state, forecast_input).sample(seed=key2)

            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)

            return (next_mean, next_cov, next_state), (marginal_mean, emission_mean_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov, initial_state)
        _, (ts_means, ts_mean_covs, ts) = lax.scan(_step, initial_params, (keys, forecast_inputs))

        return ts_means, ts_mean_covs, ts

    @jit
    def get_trans_mat(self, params, t):
        trans_mat = []
        for c_name, c_params in params:
            trans_getter = self.trans_mat_getters[c_params]
            c_trans_mat = trans_getter(c_params, t)
            trans_mat.append(c_trans_mat)
        return jsp.blockdiag(trans_mat)

    @jit
    def get_obs_mat(self, params, t):
        obs_mat = []
        for c_name, c_params in params:
            obs_getter = self.obs_mat_getters[c_name]
            c_obs_mat = obs_getter(c_params, t)
            obs_mat.append(c_obs_mat)
        return jnp.concatenate(obs_mat)

    @jit
    def get_trans_cov(self, params, t):
        trans_cov = []
        for c_name, c_params in params:
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
                 component_transition_matrices,
                 component_observation_matrices,
                 component_initial_state_priors,
                 component_transition_covariances,
                 component_transition_covariance_priors,
                 observation_covariance,
                 observation_covariance_prior,
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None):

        super().__init__(component_transition_matrices, component_observation_matrices,
                         component_initial_state_priors, component_transition_covariances,
                         component_transition_covariance_priors, cov_spars_matrices,
                         observation_regression_weights, observation_regression_weights_prior)
        # Add parameters of the observation covariance matrix.
        emission_covariance_props = ParameterProperties(
            trainable=True, constrainer=tfb.Invert(PSDToRealBijector))
        self.params.update({'emission_covariance': observation_covariance})
        self.param_props.update({'emission_covariance': emission_covariance_props})
        self.priors.update({'emission_covariance': observation_covariance_prior})

    def log_prior(self, params):
        # Compute sum of log priors of convariance matrices of the latent dynamics components,
        # as well as the log prior of parameters of the regression model (if the model has one).
        lp = super().log_prior(params)
        # Add log prior of covariance matrix of the emission model
        lp += self.priors['emission_covariance'].log_prob(params['emission_covariance'])
        return lp

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        return MVN(self.emission_matrix @ state + self.params['regression_weights'] @ inputs,
                   self.params['emission_covariance'])

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        ts_means, ts_mean_covs, ts = super().forecast(
            key, observed_time_series, num_forecast_steps, past_inputs, forecast_inputs
            )
        ts_covs = ts_mean_covs + self.params['emission_covariance']
        return ts_means, ts_covs, ts

    @jit
    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        get_trans_mat = partial(self.get_trans_mat, params=params)
        get_obs_mat = partial(self.get_obs_mat, params=params)
        get_trans_cov = partial(self.get_obs_mat, params=params)
        obs_cov = params['emission_covariance']
        emission_input_weights = params['regression_weights']
        input_dim = emission_input_weights.shape[-1]
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_covariance,
                           dynamics_matrix=get_trans_mat,
                           dynamics_input_weights=jnp.zeros((self.state_dim, input_dim)),
                           dynamics_bias=self.dynamics_bias,
                           dynamics_covariance=get_trans_cov,
                           emission_matrix=get_obs_mat,
                           emission_input_weights=emission_input_weights,
                           emission_bias=self.emission_bias,
                           emission_covariance=obs_cov)

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
                 component_transition_matrices,
                 component_observation_matrices,
                 component_initial_state_priors,
                 component_transition_covariances,
                 component_transition_covariance_priors,
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None):

        super().__init__(component_transition_matrices, component_observation_matrices,
                         component_initial_state_priors, component_transition_covariances,
                         component_transition_covariance_priors, cov_spars_matrices,
                         observation_regression_weights, observation_regression_weights_prior)

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        log_rate = self.emission_matrix @ state + self.params['regression_weights'] @ inputs
        return Poisson(rate=self._emission_constrainer(log_rate))

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        ts_means, ts_mean_covs, ts = super().forecast(
            key, observed_time_series, num_forecast_steps, past_inputs, forecast_inputs
            )
        _sample = lambda r, key: Poisson(rate=r).sample(seed=key)
        ts_samples = vmap(_sample)(ts_means, jr.split(key, num_forecast_steps))
        return ts_samples, ts_means, ts

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        get_trans_mat = partial(self.get_trans_mat, params=params)
        get_obs_mat = partial(self.get_obs_mat, params=params)
        get_trans_cov = partial(self.get_obs_mat, params=params)
        comp_cov = jsp.linalg.block_diag(*params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        return EKFParams(initial_mean=self.initial_mean,
                         initial_covariance=self.initial_covariance,
                         dynamics_function=lambda z: self.dynamics_matrix @ z,
                         dynamics_covariance=spars_cov,
                         emission_mean_function=
                         lambda z: self._emission_constrainer(self.emission_matrix @ z),
                         emission_cov_function=
                         lambda z: jnp.diag(self._emission_constrainer(self.emission_matrix @ z)))

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return cmgf_filt(params=params, emissions=emissions, inputs=inputs, num_iter=2)

    def _ssm_smoother(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return cmgf_smooth(params=params, emissions=emissions, inputs=inputs, num_iter=2)

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
