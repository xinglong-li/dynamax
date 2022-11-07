import blackjax
from collections import OrderedDict
from functools import partial
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxopt import LBFGS
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
    log_det_jac_constrain,
    flatten,
    unflatten)
from dynamax.utils import (
    pytree_stack,
    ensure_array_has_batch_dim)
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    Poisson as Pois)


class StructuralTimeSeriesSSM(SSM):
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
                 param_priors,
                 trans_mat_getters,
                 trans_cov_getters,
                 obs_mats,
                 cov_select_mats,
                 initial_distributions,
                 reg_func=None,
                 obs_distribution='Gaussian',
                 dim_covariate=0):
        self.params = params
        self.param_props = param_props
        self.param_priors = param_priors

        self.trans_mat_getters = trans_mat_getters
        self.trans_cov_getters = trans_cov_getters
        self.component_obs_mats = obs_mats
        self.cov_select_mats = cov_select_mats

        self.latent_comp_names = cov_select_mats.keys()
        self.obs_distribution = obs_distribution

        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in initial_distributions.values()])
        self.initial_cov = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in initial_distributions.values()])

        self.obs_mat = jnp.concatenate([*obs_mats.values()], axis=1)
        self.cov_select_mat = jsp.linalg.block_diag(*cov_select_mats.values())

        # Dimensions of the SSM.
        self.dim_obs, self.dim_state = self.obs_mat.shape
        self.dim_comp = self.get_trans_cov(self.params, 0).shape[0]
        self.dim_covariate = dim_covariate

        # Pick out the regression component if there is one.
        if reg_func is not None:
            # The regression component is always the last component if there is one.
            self.reg_name = list(params.keys())[-1]
            self.regression = reg_func

    @property
    def emission_shape(self):
        return (self.dim_obs,)

    @property
    def covariates_shape(self):
        return (self.dim_covariate,)

    def log_prior(self, params):
        """Log prior probability of parameters.
        """
        lp = 0.
        for c_name, c_priors in self.param_priors.items():
            for p_name, p_pri in c_priors.items():
                lp += p_pri.log_prob(params[c_name][p_name])
        return lp

    # Instantiate distributions of the SSM model
    def initial_distribution(self):
        """Distribution of the initial state of the SSM form of the STS model.
        """
        return MVN(self.initial_mean, self.initial_cov)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
           multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, inputs):
        """Depends on the distribution family of the observation.
        """
        if self.obs_distribution == 'Gaussian':
            return MVN(self.obs_mat @ state + inputs, self.params['obs_model']['cov'])
        elif self.obs_distribution == 'Poisson':
            unc_rates = self.obs_mat @ state + inputs
            return Pois(rate=self._emission_constrainer(unc_rates))

    def sample(self, params, num_timesteps, covariates=None, key=jr.PRNGKey(0)):
        """Sample a sequence of latent states and emissions with the given parameters.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros((num_timesteps, self.dim_obs))

        get_trans_mat = partial(self.get_trans_mat, params)
        get_trans_cov = partial(self.get_trans_cov, params)

        def _step(prev_state, args):
            key, input, t = args
            key1, key2 = jr.split(key, 2)
            _next_state = get_trans_mat(t) @ prev_state
            next_state = _next_state + self.cov_select_mat @ MVN(
                jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)
            prev_obs = self.emission_distribution(prev_state, input).sample(seed=key2)
            return next_state, (prev_state, prev_obs)

        # Sample the initial state.
        key1, key2 = jr.split(key, 2)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the following emissions and states.
        key2s = jr.split(key2, num_timesteps)
        _, (states, time_series) = lax.scan(
            _step, initial_state, (key2s, inputs, jnp.arange(num_timesteps)))
        return states, time_series

    def marginal_log_prob(self, params, obs_time_series, covariates=None):
        """Compute log marginal likelihood of observations.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the model to SSM.
        ssm_params = self._to_ssm_params(params)

        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=inputs)
        return filtered_posterior.marginal_loglik

    def posterior_sample(self, params, obs_time_series, covariates=None, key=jr.PRNGKey(0)):
        """Posterior sample.
        """
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)

        # Sample latent state.
        key1, key2 = jr.split(key, 2)
        ll, states = self._ssm_posterior_sample(ssm_params, obs_time_series, inputs, key1)

        # Sample observations.
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
        component_pos = OrderedDict()
        if covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            inputs = self.regression(params[self.reg_name], covariates)
            component_pos[self.reg_name] = {'pos_mean': jnp.squeeze(inputs),
                                            'pos_cov': jnp.zeros_like(obs_time_series)}
        else:
            inputs = jnp.zeros(obs_time_series.shape)

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)

        # Infer the posterior of the joint SSM model.
        pos = self._ssm_smoother(ssm_params, obs_time_series, inputs)
        mu_pos = pos.smoothed_means
        var_pos = pos.smoothed_covariances

        # Decompose by latent component.
        _loc = 0
        for c, obs_mat in self.component_obs_mats.items():
            # Extract posterior mean and covariances of each component
            # from the joint latent state.
            c_dim = obs_mat.shape[1]
            c_mean = mu_pos[:, _loc:_loc+c_dim]
            c_cov = var_pos[:, _loc:_loc+c_dim, _loc:_loc+c_dim]
            # Posterior emission of the single component.
            c_obs_mean = vmap(jnp.matmul, (None, 0))(obs_mat, c_mean)
            c_obs_constrained_mean = self._emission_constrainer(c_obs_mean)
            c_obs_cov = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_cov, obs_mat)
            component_pos[c] = {'pos_mean': c_obs_constrained_mean,
                                'pos_cov': c_obs_cov}
            _loc += c_dim
        return component_pos

    def fit_vi(self,
               initial_params,
               param_props,
               key,
               sample_size,
               emissions,
               covariates=None,
               M=100):
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
        lbfgs = LBFGS(fun=objective)
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

    def fit_hmc(self,
                initial_params,
                param_props,
                key,
                num_samples,
                emissions,
                covariates=None,
                warmup_steps=100,
                verbose=True):
        """Sample parameters of the model using HMC.
        """
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
        warmup = blackjax.window_adaptation(blackjax.nuts, unnorm_log_pos, num_steps=warmup_steps)
        init_key, key = jr.split(key)
        hmc_initial_state, hmc_kernel, _ = warmup.run(init_key, initial_unc_params)

        @jit
        def hmc_step(hmc_state, step_key):
            next_hmc_state, _ = hmc_kernel(step_key, hmc_state)
            params = from_unconstrained(hmc_state.position, fixed_params, param_props)
            return next_hmc_state, params

        # Start sampling.
        log_probs = []
        samples = []
        hmc_state = hmc_initial_state
        # pbar = trange(num_samples) if verbose else range(num_samples)
        pbar = range(num_samples)
        for _ in pbar:
            step_key, key = jr.split(key)
            hmc_state, params = hmc_step(hmc_state, step_key)
            log_probs.append(-hmc_state.potential_energy)
            samples.append(params)

        # Combine the samples into a single pytree
        return pytree_stack(samples), jnp.array(log_probs)

    def forecast(self,
                 params,
                 obs_time_series,
                 num_forecast_steps,
                 past_covariates=None,
                 forecast_covariates=None,
                 key=jr.PRNGKey(0)):
        """Forecast the time series.
        """
        if forecast_covariates is not None:
            # If there is a regression component, set the inputs of the emission model
            # of the SSM as the fitted value of the regression component.
            past_inputs = self.regression(params[self.reg_name], past_covariates)
            forecast_inputs = self.regression(params[self.reg_name], forecast_covariates)
        else:
            past_inputs = jnp.zeros(obs_time_series.shape)
            forecast_inputs = jnp.zeros((num_forecast_steps, self.dim_obs))

        # Convert the STS model to SSM.
        ssm_params = self._to_ssm_params(params)
        get_trans_mat = partial(self.get_trans_mat, params)
        get_trans_cov = partial(self.get_trans_cov, params)

        # Filtering the observed time series to initialize the forecast
        filtered_posterior = self._ssm_filter(
            params=ssm_params, emissions=obs_time_series, inputs=past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        def _step(current_states, args):
            key, forecast_input, t = args
            key1, key2 = jr.split(key)
            cur_mean, cur_cov, cur_state = current_states

            # Observation of the previous time step.
            obs_mean_unc = self.obs_mat @ cur_mean + forecast_input
            obs_mean = self._emission_constrainer(obs_mean_unc)
            obs_cov = self.obs_mat @ cur_cov @ self.obs_mat.T + params['obs_model']['cov']
            obs = self.emission_distribution(cur_state, forecast_input).sample(seed=key2)

            # Predict the latent state of the next time step.
            next_mean = get_trans_mat(t) @ cur_mean
            next_cov = get_trans_mat(t) @ cur_cov @ get_trans_mat(t).T\
                + self.cov_select_mat @ get_trans_cov(t) @ self.cov_select_mat.T
            next_state = get_trans_mat(t) @ cur_state\
                + self.cov_select_mat @ MVN(jnp.zeros(self.dim_comp), get_trans_cov(t)).sample(seed=key1)
            return (next_mean, next_cov, next_state), (obs_mean, obs_cov, obs)

        # The first time step of forecast.
        t0 = obs_time_series.shape[0]
        initial_mean = get_trans_mat(t0) @ filtered_mean[-1]
        initial_cov = get_trans_mat(t0) @ filtered_cov[-1] @ get_trans_mat(t0).T\
            + self.cov_select_mat @ get_trans_cov(t0) @ self.cov_select_mat.T
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)
        initial_states = (initial_mean, initial_cov, initial_state)

        # Forecast the following up steps.
        carrys = (jr.split(key, num_forecast_steps),
                  forecast_inputs,
                  t0 + 1 + jnp.arange(num_forecast_steps))
        _, (ts_means, ts_mean_covs, ts) = lax.scan(_step, initial_states, carrys)

        return ts_means, ts_mean_covs, ts

    def one_step_predict(self, params, obs_time_series, covariates=None):
        """One step forward prediction.
           This is a by product of the Kalman filter.
        """
        raise NotImplementedError

    def get_trans_mat(self, params, t):
        """Evaluate the transition matrix of the latent state at time step t,
           conditioned on parameters of the model.
        """
        trans_mat = []
        for c_name in self.latent_comp_names:
            # Obtain the transition matrix of each single latent component.
            trans_getter = self.trans_mat_getters[c_name]
            c_trans_mat = trans_getter(params[c_name], t)
            trans_mat.append(c_trans_mat)
        return jsp.linalg.block_diag(*trans_mat)

    def get_trans_cov(self, params, t):
        """Evaluate the covariance of the latent dynamics at time step t,
           conditioned on parameters of the model.
        """
        trans_cov = []
        for c_name in self.latent_comp_names:
            # Obtain the covariance of each single latent component.
            cov_getter = self.trans_cov_getters[c_name]
            c_trans_cov = cov_getter(params[c_name], t)
            trans_cov.append(c_trans_cov)
        return jsp.linalg.block_diag(*trans_cov)

    def _to_ssm_params(self, params):
        """Convert the STS model into the form of the corresponding SSM model.
        """
        get_trans_mat = partial(self.get_trans_mat, params)
        get_sparse_cov = lambda t:\
            self.cov_select_mat @ self.get_trans_cov(params, t) @ self.cov_select_mat.T
        if self.obs_distribution == 'Gaussian':
            return LGSSMParams(initial_mean=self.initial_mean,
                               initial_covariance=self.initial_cov,
                               dynamics_matrix=get_trans_mat,
                               dynamics_input_weights=jnp.zeros((self.dim_state, 1)),
                               dynamics_bias=jnp.zeros(self.dim_state),
                               dynamics_covariance=get_sparse_cov,
                               emission_matrix=self.obs_mat,
                               emission_input_weights=jnp.eye(self.dim_obs),
                               emission_bias=jnp.zeros(self.dim_obs),
                               emission_covariance=params['obs_model']['cov'])
        elif self.obs_distribution == 'Poisson':
            # Current formulation of the dynamics function cannot depends on t
            trans_mat = get_trans_mat(0)
            sparse_cov = get_sparse_cov(0)
            return GGSSMParams(initial_mean=self.initial_mean,
                               initial_covariance=self.initial_cov,
                               dynamics_function=lambda z: trans_mat @ z,
                               dynamics_covariance=sparse_cov,
                               emission_mean_function=
                                   lambda z: self._emission_constrainer(self.obs_mat @ z),
                               emission_cov_function=
                                   lambda z: jnp.diag(self._emission_constrainer(self.obs_mat @ z)),
                               emission_dist=lambda mu, _: Pois(log_rate=jnp.log(mu)))

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model.
        """
        if self.obs_distribution == 'Gaussian':
            return lgssm_filter(params=params, emissions=emissions, inputs=inputs)
        elif self.obs_distribution == 'Poisson':
            return cmgf_filt(params=params, inf_params=EKFIntegrals(), emissions=emissions,
                             inputs=inputs, num_iter=2)

    def _ssm_smoother(self, params, emissions, inputs):
        """The smoother of the corresponding SSM model
        """
        if self.obs_distribution == 'Gaussian':
            return lgssm_smoother(params=params, emissions=emissions, inputs=inputs)
        elif self.obs_distribution == 'Poisson':
            return cmgf_smooth(params=params, inf_params=EKFIntegrals(), emissions=emissions,
                               inputs=inputs, num_iter=2)

    def _ssm_posterior_sample(self, ssm_params, obs_time_series, inputs, key):
        """The posterior sampler of the corresponding SSM model
        """
        if self.obs_distribution == 'Gaussian':
            return lgssm_posterior_sample(
                rng=key, params=ssm_params, emissions=obs_time_series, inputs=inputs)
        elif self.obs_distribution == 'Poisson':
            # Currently the posterior_sample for STS model with Poisson likelihood
            # simply returns the filtered means.
            return self._ssm_filter(ssm_params, obs_time_series, inputs)

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space.
        """
        if self.obs_distribution == 'Gaussian':
            return emission
        elif self.obs_distribution == 'Poisson':
            return jnp.exp(emission)
