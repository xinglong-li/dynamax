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
                 cov_select_mat):
        self.params = params
        self.param_props = param_props
        self.param_priors = priors

        self.component_init_dists = initial_distributions
        self.cov_select_mat = cov_select_mat

        self.trans_mat_getters = OrderedDict()
        self.obs_mat_getters = OrderedDict()
        self.trans_cov_getters = OrderedDict()

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
        raise NotImplementedError

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
           multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state):
        """Depends on the distribution family of the observation.
        """
        raise NotImplementedError

    @jit
    def sample(self, params, key, num_timesteps):
        """Sample a sequence of latent states and emissions with the given parameters.

        Since the regression is contained in the regression component,
        so there is no covariates term under the STS framework.
        """
        initial_dists = self.component_init_dists
        get_trans_mat = partial(self.get_trans_mat, params=params)
        get_trans_cov = partial(self.get_trans_cov, params=params)
        cov_select_mat = self.cov_select_mat
        dim_comp = get_trans_cov(0).shape[-1]

        def _step(prev_state, args):
            key, t = args
            key1, key2 = jr.split(key, 2)
            next_state = get_trans_mat(t) @ prev_state
            next_state = next_state + cov_select_mat @ MVN(jnp.zeros(dim_comp),
                                                           get_trans_cov(t)).sample(seed=key1)
            emission = self.emission_distribution(prev_state).sample(seed=key2)
            return next_state, (prev_state, emission)

        # Sample the initial state
        key1, key2 = jr.split(key, 2)
        key1s = jr.split(key1, len(initial_dists))
        initial_state = jnp.concatenate([c_dist.sample(seed=key)
                                         for c_dist, key in (initial_dists, key1s)])

        # Sample the remaining emissions and states
        key2s = jr.split(key2, num_timesteps)
        _, (states, emissions) = lax.scan(_step, initial_state, (key2s, jnp.arange(num_timesteps)))

        return states, emissions

    @jit
    def marginal_log_prob(self, params, obs_time_series):
        """Compute log marginal likelihood of observations."""
        ssm_params = self._to_ssm_params(params)
        filtered_posterior = self._ssm_filter(params=ssm_params, emissions=obs_time_series)
        return filtered_posterior.marginal_loglik

    @jit
    def posterior_sample(self, params, key, obs_time_series):
        key1, key2 = jr.split(key, 2)
        num_timesteps, dim_obs = obs_time_series.shape
        ssm_params = self._to_ssm_params(params)
        obs_mats = vmap(self.get_obs_mat, (None, 0))(params, jnp.arange(num_timesteps))

        ll, states = self._ssm_posterior_sample(key1, ssm_params, obs_time_series)
        obs_means = vmap(jnp.matmul)(obs_mats, obs_time_series)
        obs_means = self._emission_constrainer(obs_means)
        key2s = jr.split(key2, num_timesteps)
        emission_sampler = lambda state, key: self.emission_distribution(state).sample(seed=key)
        obs = vmap(emission_sampler)(states, key2s)
        return obs_means, obs

    def component_posterior(self, obs_time_series):
        """Smoothing by component
        """
        # Compute the posterior of the joint SSM model
        component_pos = OrderedDict()
        ssm_params = self._to_ssm_params(self.params)
        pos = self._ssm_smoother(ssm_params, obs_time_series)
        mu_pos = pos.smoothed_means
        var_pos = pos.smoothed_covariances

        # Decompose by component
        _loc = 0
        for c, emission_matrix in self.component_emission_matrices.items():
            c_dim = emission_matrix.shape[-1]
            c_mu = mu_pos[:, _loc:_loc+c_dim]
            c_var = var_pos[:, _loc:_loc+c_dim, _loc:_loc+c_dim]
            c_emission_mu = vmap(lambda s, m: m @ s, (0, None))(c_mu, emission_matrix)
            c_emission_constrained_mu = self._emission_constrainer(c_emission_mu)
            c_emission_var = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_var, emission_matrix)
            component_pos[c] = (c_emission_constrained_mu, c_emission_var)
            _loc += c_dim

        return component_pos

    def fit_hmc(self,
                key,
                sample_size,
                emissions,
                inputs=None,
                warmup_steps=500,
                num_integration_steps=20):

        def unnorm_log_pos(trainable_unc_params):
            params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(trainable_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(params, emissions, inputs)
            lp = log_pri + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptations
        hmc_initial_position, fixed_params = to_unconstrained(self.params, self.param_props)
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            unnorm_log_pos,
                                            num_steps=warmup_steps,
                                            num_integration_steps=num_integration_steps)
        hmc_initial_state, hmc_kernel, _ = warmup.run(key, hmc_initial_position)

        @jit
        def _step(current_state, rng_key):
            next_state, _ = hmc_kernel(rng_key, current_state)
            unc_sample = next_state.position
            return next_state, unc_sample

        keys = iter(jr.split(key, sample_size))
        param_samples = []
        current_state = hmc_initial_state
        for _ in trange(sample_size):
            current_state, unc_sample = _step(current_state, next(keys))
            sample = from_unconstrained(unc_sample, fixed_params, self.param_props)
            param_samples.append(sample)

        param_samples = tree_map(lambda x, *y: jnp.array([x] + [i for i in y]),
                                 param_samples[0], *param_samples[1:])
        return param_samples

    def fit_vi(self, key, sample_size, emissions, inputs=None, M=100):
        """
        ADVI approximate the posterior distribtuion p of unconstrained global parameters
        with factorized multivatriate normal distribution:
        q = \prod_{k=1}^{K} q_k(mu_k, sigma_k),
        where K is dimension of p.

        The hyper-parameters of q to be optimized over are (mu_k, log_sigma_k))_{k=1}^{K}.

        The trick of reparameterization is employed to reduce the variance of SGD,
        which is achieved by written KL(q || p) as expectation over standard normal distribution
        so a sample from q is obstained by
        s = z * exp(log_sigma_k) + mu_k,
        where z is a sample from the standard multivarate normal distribtion.

        This implementation of ADVI uses fixed samples from q during fitting, instead of
        updating samples from q in each iteration, as in SGD.
        So the second order fixed optimization algorithm L-BFGS is used.

        Args:
            sample_size (int): number of samples to be returned from the fitted approxiamtion q.
            M (int): number of fixed samples from q used in evaluation of ELBO.

        Returns:
            Samples from the approximate posterior q
        """
        key0, key1 = jr.split(key)
        model_unc_params, fixed_params = to_unconstrained(self.params, self.param_props)
        params_flat, params_structure = flatten(model_unc_params)
        vi_dim = len(params_flat)

        std_normal = MVNDiag(jnp.zeros(vi_dim), jnp.ones(vi_dim))
        std_samples = std_normal.sample(seed=key0, sample_shape=(M,))
        std_samples = vmap(unflatten, (None, 0))(params_structure, std_samples)

        @jit
        def unnorm_log_pos(unc_params):
            """Unnormalzied log posterior of global parameters."""

            params = from_unconstrained(unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(params, emissions, inputs)
            lp = log_pri + batch_lls.sum()
            return lp

        @jit
        def _samp_elbo(vi_params, std_samp):
            """Evaluate ELBO at one sample from the approximate distribution q.
            """
            vi_means, vi_log_sigmas = vi_params
            # unc_params_flat = vi_means + std_samp * jnp.exp(vi_log_sigmas)
            # unc_params = unflatten(params_structure, unc_params_flat)
            # With reparameterization, entropy of q evaluated at z is
            # sum(hyper_log_sigma) plus some constant depending only on z.
            _params = tree_map(lambda x, log_sig: x * jnp.exp(log_sig), std_samp, vi_log_sigmas)
            unc_params = tree_map(lambda x, mu: x + mu, _params, vi_means)
            q_entropy = flatten(vi_log_sigmas)[0].sum()
            return q_entropy + unnorm_log_pos(unc_params)

        objective = lambda x: -vmap(_samp_elbo, (None, 0))(x, std_samples).mean()

        # Fit ADVI with LBFGS algorithm
        initial_vi_means = model_unc_params
        initial_vi_log_sigmas = unflatten(params_structure, jnp.zeros(vi_dim))
        initial_vi_params = (initial_vi_means, initial_vi_log_sigmas)
        lbfgs = LBFGS(maxiter=1000, fun=objective, tol=1e-3, stepsize=1e-3, jit=True)
        (vi_means, vi_log_sigmas), _info = lbfgs.run(initial_vi_params)

        # Sample from the learned approximate posterior q
        _samples = std_normal.sample(seed=key1, sample_shape=(sample_size,))
        _vi_means = flatten(vi_means)[0]
        _vi_log_sigmas = flatten(vi_log_sigmas)[0]
        vi_samples_flat = _vi_means + _samples * jnp.exp(_vi_log_sigmas)
        vi_unc_samples = vmap(unflatten, (None, 0))(params_structure, vi_samples_flat)
        vi_samples = vmap(from_unconstrained, (0, None, None))(
            vi_unc_samples, fixed_params, self.param_props)

        return vi_samples

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        """Forecast the time series"""

        if forecast_inputs is None:
            forecast_inputs = jnp.zeros((num_forecast_steps, 0))
        weights = self.params['regression_weights']
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        dim_comp = comp_cov.shape[-1]

        # Filtering the observed time series to initialize the forecast
        ssm_params = self._to_ssm_params(self.params)
        filtered_posterior = self._ssm_filter(params=ssm_params,
                                              emissions=observed_time_series,
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
