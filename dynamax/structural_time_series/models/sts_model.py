from collections import OrderedDict
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from dynamax.distributions import InverseWishart as IW
from dynamax.parameters import ParameterProperties as Prop
from dynamax.structural_time_series.models.sts_ssm import GaussianSSM, PoissonSSM
from dynamax.structural_time_series.models.sts_components import *
from dynamax.utils import PSDToRealBijector
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb


RealToPSD = tfb.Invert(PSDToRealBijector)


class StructuralTimeSeries():
    """The class of the Bayesian structural time series (STS) model:

    y_t = H_t @ z_t + \err_t,   \err_t \sim N(0, \Sigma_t) 
    z_{t+1} = F_t @ z_t + R_t @ \eta_t, eta_t \sim N(0, Q_t)

    H_t: emission matrix
    F_t: transition matrix of the dynamics
    R_t: subset of clumns of base vector I, and is called'selection matrix'
    Q_t: nonsingular covariance matrix of the latent state

    Construct a structural time series (STS) model from a list of components

    Args:
        components: list of components
        observation_covariance:
        observation_covariance_prior: InverseWishart prior for the observation covariance matrix
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        name (str): name of the STS model
    """

    def __init__(self,
                 components,
                 obs_time_series,
                 obs_distribution_family='Gaussian',
                 obs_cov_props=None,
                 obs_cov_prior=None,
                 obs_cov=None,
                 covariates=None,
                 name='sts_model'):

        names = [c.name for c in components]
        assert len(set(names)) == len(names), "Components should not share the same name."
        assert obs_distribution_family in ['Gaussian', 'Poisson'],\
            "The distribution of observations must be Gaussian or Poisson."

        self.name = name
        self.dim_obs = obs_time_series.shape[-1]
        self.obs_family = obs_distribution_family

        # Initialize paramters using the scale of observed time series
        regression = None
        obs_scale = jnp.std(jnp.abs(jnp.diff(obs_time_series, axis=0)), axis=0)
        for c in components:
            if isinstance(c, STSRegression):
                regression = c
                regression.initialize(covariates, obs_time_series)
                residuals = regression.fit(regression.params, covariates)
                obs_scale = jnp.std(jnp.abs(jnp.diff(residuals, axis=0)), axis=0)
        for c in components:
            if not isinstance(c, STSRegression):
                c.initialize_params(obs_time_series[0], obs_scale)

        # Aggeragate components
        self.initial_distributions = OrderedDict()
        self.param_props = OrderedDict()
        self.param_priors = OrderedDict()
        self.params = OrderedDict()
        self.trans_mat_getters = OrderedDict()
        self.trans_cov_getters = OrderedDict()
        self.obs_mats = OrderedDict()
        self.cov_select_mats = OrderedDict()

        for c in components:
            if not isinstance(c, STSRegression):
                self.initial_distributions[c.name] = c.initial_distribution
                self.param_props[c.name] = c.param_props
                self.param_priors[c.name] = c.param_priors
                self.params[c.name] = c.params
                self.trans_mat_getters[c.name] = c.get_trans_mat
                self.trans_cov_getters[c.name] = c.get_trans_cov
                self.obs_mats[c.name] = c.obs_mat
                self.cov_select_mats[c.name] = c.cov_select_mat

        if self.obs_family == 'Gaussian':
            if obs_cov_props is None:
                obs_cov_props = Prop(trainable=True, constrainer=RealToPSD)
            if obs_cov_prior is None:
                obs_cov_prior = IW(df=self.dim_obs, scale=jnp.eye(self.dim_obs))
            if obs_cov is None:
                obs_cov = obs_cov_prior.mode()
            self.param_props['obs_model'] = OrderedDict({'cov': obs_cov_props})
            self.param_priors['obs_model'] = OrderedDict({'cov': obs_cov_prior})
            self.params['obs_model'] = OrderedDict({'cov': obs_cov})

        # Always put the regression term at the last position of the OrderedDict.
        if regression is not None:
            self.param_props[regression.name] = regression.param_props
            self.param_priors[regression.name] = regression.param_priors
            self.params[regression.name] = regression.params
            self.reg_func = regression.fit
        else:
            self.reg_func = None

    def as_ssm(self):
        """Formulate the STS model as a linear Gaussian state space model:

        p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
        p(y_t | z_t) =
        p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})

        F_t, H_t are fixed known matrices,
        the convariance matrices, Q and R, are random variables to be learned,
        the regression coefficient matrix B is also unknown random matrix
        if the STS model includes an regression component
        """
        if self.obs_family == 'Gaussian':
            sts_ssm = GaussianSSM(
                self.param_props, self.param_priors, self.params,
                self.trans_mat_getters, self.trans_cov_getters,
                self.obs_mats, self.cov_select_mats, self.initial_distributions, self.reg_func)
        elif self.obs_family == 'Poisson':
            sts_ssm = PoissonSSM(
                self.params, self.param_props, self.param_priors, self.get_trans_mat, self.get_obs_mat,
                self.get_trans_cov, self.initial_mean, self.initial_cov, self.cov_select_mat)
        return sts_ssm

    def decompose_by_component(self, observed_time_series, inputs=None,
                               sts_params=None, num_post_samples=100, key=jr.PRNGKey(0)):
        """Decompose the STS model into components and return the means and variances
           of the marginal posterior of each component.

           The marginal posterior of each component is obtained by averaging over
           conditional posteriors of that component using Kalman smoother conditioned
           on the sts_params. Each sts_params is a posterior sample of the STS model
           conditioned on observed_time_series.

           The marginal posterior mean and variance is computed using the formula
           E[X] = E[E[X|Y]],
           Var(Y) = E[Var(X|Y)] + Var[E[X|Y]],
           which holds for any random variables X and Y.

        Args:
            observed_time_series (_type_): _description_
            inputs (_type_, optional): _description_. Defaults to None.
            sts_params (_type_, optional): Posteriror samples of STS parameters.
                If not given, 'num_posterior_samples' of STS parameters will be
                sampled using self.fit_hmc.
            num_post_samples (int, optional): Number of posterior samples of STS
                parameters to be sampled using self.fit_hmc if sts_params=None.

        Returns:
            component_dists: (OrderedDict) each item is a tuple of means and variances
                              of one component.
        """
        component_dists = OrderedDict()

        # Sample parameters from the posterior if parameters is not given
        if sts_params is None:
            sts_ssm = self.as_ssm()
            sts_params = sts_ssm.fit_hmc(key, num_post_samples, observed_time_series, inputs)

        @jit
        def decomp_poisson(sts_param):
            """Decompose one STS model if the observations follow Poisson distributions.
            """
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 sts_param['dynamics_covariances'],
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 sts_param['regression_weights'],
                                 self.observation_regression_weights_prior)
            return sts_ssm.component_posterior(observed_time_series, inputs)

        @jit
        def decomp_gaussian(sts_param):
            """Decompose one STS model if the observations follow Gaussian distributions.
            """
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  sts_param['dynamics_covariances'],
                                  self.transition_covariance_priors,
                                  sts_param['emission_covariance'],
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  sts_param['regression_weights'],
                                  self.observation_regression_weights_prior)
            return sts_ssm.component_posterior(observed_time_series, inputs)

        # Obtain the smoothed posterior for each component given the parameters
        if self.obs_family == 'Gaussian':
            component_conditional_pos = vmap(decomp_gaussian)(sts_params)
        elif self.obs_family == 'Poisson':
            component_conditional_pos = vmap(decomp_poisson)(sts_params)

        # Obtain the marginal posterior
        for c, pos in component_conditional_pos.items():
            mus = pos[0]
            vars = pos[1]
            # Use the formula: E[X] = E[E[X|Y]]
            mu_series = mus.mean(axis=0)
            # Use the formula: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
            var_series = jnp.mean(vars, axis=0)[..., 0] + jnp.var(mus, axis=0)
            component_dists[c] = (mu_series, var_series)

        return component_dists

    def sample(self, key, num_timesteps, inputs=None):
        """Given parameters, sample latent states and corresponding observed time series.
        """
        sts_ssm = self.as_ssm()
        states, timeseries = sts_ssm.sample(key, num_timesteps, inputs)
        return timeseries

    def marginal_log_prob(self, observed_time_series, inputs=None):
        sts_ssm = self.as_ssm()
        return sts_ssm.marginal_log_prob(sts_ssm.params, observed_time_series, inputs)

    def posterior_sample(self, key, observed_time_series, sts_params, inputs=None):
        @jit
        def single_sample_poisson(sts_param):
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 sts_param['dynamics_covariances'],
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 sts_param['regression_weights'],
                                 self.observation_regression_weights_prior
                                 )
            ts_means, ts = sts_ssm.posterior_sample(key, observed_time_series, inputs)
            return [ts_means, ts]

        @jit
        def single_sample_gaussian(sts_param):
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  sts_param['dynamics_covariances'],
                                  self.transition_covariance_priors,
                                  sts_param['emission_covariance'],
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  sts_param['regression_weights'],
                                  self.observation_regression_weights_prior
                                  )
            ts_means, ts = sts_ssm.posterior_sample(key, observed_time_series, inputs)
            return [ts_means, ts]

        if self.obs_family == 'Gaussian':
            samples = vmap(single_sample_gaussian)(sts_params)
        elif self.obs_family == 'Poisson':
            samples = vmap(single_sample_poisson)(sts_params)

        return {'means': samples[0], 'observations': samples[1]}

    def fit_hmc(self, key, sample_size, observed_time_series, inputs=None,
                warmup_steps=500, num_integration_steps=30):
        """Sample parameters of the STS model from their posterior distributions.

        Parameters of the STS model includes:
            covariance matrix of each component,
            regression coefficient matrix (if the model has inputs and a regression component)
            covariance matrix of observations (if observations follow Gaussian distribution)
        """
        # Initialize via fit MLE
        sts_ssm = self.as_ssm()
        param_samps = sts_ssm.fit_hmc(key, sample_size, observed_time_series, inputs,
                                      warmup_steps, num_integration_steps)
        return param_samps

    def fit_mle(self, obs_time_series, covariates=None, num_steps=1000,
                initial_params=None, optimizer=optax.adam(1e-1), key=jr.PRNGKey(0)):
        """Maximum likelihood estimate of parameters of the STS model
        """
        sts_ssm = self.as_ssm()

        batch_obs = jnp.array([obs_time_series])
        if covariates is not None:
            covariates = jnp.array([covariates])
        curr_params = sts_ssm.params if initial_params is None else initial_params
        param_props = sts_ssm.param_props

        optimal_params, losses = sts_ssm.fit_sgd(
            curr_params, param_props, batch_obs, num_epochs=num_steps,
            key=key, covariates=covariates, optimizer=optimizer)

        return optimal_params, losses

    def forecast(self, key, observed_time_series, sts_params, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        @jit
        def single_forecast_gaussian(sts_param):
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  sts_param['dynamics_covariances'],
                                  self.transition_covariance_priors,
                                  sts_param['emission_covariance'],
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  sts_param['regression_weights'],
                                  self.observation_regression_weights_prior
                                  )
            means, covs, ts = sts_ssm.forecast(key, observed_time_series, num_forecast_steps,
                                               past_inputs, forecast_inputs)
            return [means, covs, ts]

        @jit
        def single_forecast_poisson(sts_param):
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 sts_param['dynamics_covariances'],
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 sts_param['regression_weights'],
                                 self.observation_regression_weights_prior
                                 )
            means, covs, ts = sts_ssm.forecast(key, observed_time_series, num_forecast_steps,
                                               past_inputs, forecast_inputs)
            return [means, covs, ts]

        if self.obs_family == 'Gaussian':
            forecasts = vmap(single_forecast_gaussian)(sts_params)
        elif self.obs_family == 'Poisson':
            forecasts = vmap(single_forecast_poisson)(sts_params)

        return {'means': forecasts[0], 'covariances': forecasts[1], 'observations': forecasts[2]}
