from functools import partial
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import tree_map
from dynamax.abstractions import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample, LGSSMParams
from dynamax.parameters import ParameterProperties
from dynamax.utils import PSDToRealBijector
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

tfd = tfp.distributions
tfb = tfp.bijectors

_zeros_if_none = lambda x, shp: x if x is not None else jnp.zeros(shp)

class LinearGaussianSSM(SSM):
    """
    Linear Gaussian State Space Model.

    The model is defined as follows:

    p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
    p(y_t | z_t) = N(y_t | H_t z_t + D_t u_t + d_t, R_t)
    p(z_1) = N(z_1 | m, S)

    where

    z_t = hidden variables of size `state_dim`,
    y_t = observed variables of size `emission_dim`
    u_t = input covariates of size `covariate_dim` (defaults to 0)

    The parameters of the model are stored in a separate dictionary, as follows:
    F = params["dynamics"]["weights"]
    Q = params["dynamics"]["cov"]
    H = params["emission"]["weights"]
    R = params["emissions"]["cov]
    m = params["init"]["mean"]
    S = params["init"]["cov"]
    Optional parameters (default to 0)
    B = params["dynamics"]["input_weights"]
    b = params["dynamics"]["bias"]
    D = params["emission"]["input_weights"]
    d = params["emission"]["bias"]

    You can create these parameters manually, or by calling `initialize`.
    """
    def __init__(self,
                 state_dim,
                 emission_dim,
                 covariate_dim=0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.covariate_dim = covariate_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return (self.covariate_dim,) if self.covariate_dim > 0 else None

    def initialize(self,
                   key=jr.PRNGKey(0),
                   initial_mean=None,
                   initial_covariance=None,
                   dynamics_weights=None,
                   dynamics_bias=None,
                   dynamics_input_weights=None,
                   dynamics_covariance=None,
                   emission_weights=None,
                   emission_bias=None,
                   emission_input_weights=None,
                   emission_covariance=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled randomly and/or set to reasonable defaults.

        Note: in the future we may support more initialization schemes.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_mean (array, optional): manually specified initial mean. Defaults to None.
            initial_covariance (array, optional): manually specified initial covariance. Defaults to None.
            dynamics_weights (array, optional): manually specified dynamics weights. Defaults to None.
            dynamics_bias (array, optional): manually specified dynamics bias. Defaults to None.
            dynamics_input_weights (array, optional): manually specified dynamics input weights. Defaults to None.
            dynamics_covariance (array, optional): manually specified dynamics covariance. Defaults to None.
            emission_weights (array, optional): manually specified emission weights. Defaults to None.
            emission_bias (array, optional): manually specified emission bias. Defaults to None.
            emission_input_weights (array, optional): manually specified emission input weights. Defaults to None.
            emission_covariance (array, optional): manually specified emission covariance. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        _initial_mean = jnp.zeros(self.state_dim)
        _initial_covariance = jnp.eye(self.state_dim)
        _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros((self.state_dim, self.covariate_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)
        _emission_weights = jr.normal(key, (self.emission_dim, self.state_dim))
        _emission_input_weights = jnp.zeros((self.emission_dim, self.covariate_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = dict(
            initial=dict(mean=default(initial_mean, _initial_mean),
                         cov=default(initial_covariance, _initial_covariance)),
            dynamics=dict(weights=default(dynamics_weights, _dynamics_weights),
                          bias=default(dynamics_bias, _dynamics_bias),
                          input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                          cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=dict(weights=default(emission_weights, _emission_weights),
                           bias=default(emission_bias, _emission_bias),
                           input_weights=default(emission_input_weights, _emission_input_weights),
                           cov=default(emission_covariance, _emission_covariance))
        )

        param_props = dict(
            initial=dict(mean=ParameterProperties(),
                         cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            dynamics=dict(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            emissions=dict(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        )
        return params, param_props

    def initial_distribution(self, params, covariates=None):
        return MVN(params["initial"]["mean"], params["initial"]["cov"])

    def transition_distribution(self, params, state, covariates=None):
        inputs = covariates if covariates is not None else jnp.zeros(self.covariate_dim)
        mean = params["dynamics"]["weights"] @ state + params["dynamics"]["input_weights"] @ inputs
        if self.has_dynamics_bias:
            mean += params["dynamics"]["bias"]
        return MVN(mean, params["dynamics"]["cov"])

    def emission_distribution(self, params, state, covariates=None):
        inputs = covariates if covariates is not None else jnp.zeros(self.covariate_dim)
        mean = params["emissions"]["weights"] @ state + params["emissions"]["input_weights"] @ inputs
        if self.has_emissions_bias:
            mean += params["emissions"]["bias"]
        return MVN(mean, params["emissions"]["cov"])

    def _make_inference_args(self, params):
        """Convert params dict to LGSSMParams container replacing Nones if necessary."""
        dyn_bias = _zeros_if_none(params["dynamics"]["bias"], self.state_dim)
        ems_bias = _zeros_if_none(params["emissions"]["bias"], self.emission_dim)
        return LGSSMParams(initial_mean=params["initial"]["mean"],
                           initial_covariance=params["initial"]["cov"],
                           dynamics_matrix=params["dynamics"]["weights"],
                           dynamics_input_weights=params["dynamics"]["input_weights"],
                           dynamics_bias=dyn_bias,
                           dynamics_covariance=params["dynamics"]["cov"],
                           emission_matrix=params["emissions"]["weights"],
                           emission_input_weights=params["emissions"]["input_weights"],
                           emission_bias=ems_bias,
                           emission_covariance=params["emissions"]["cov"])

    def log_prior(self, params):
        """Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

    def marginal_log_prob(self, params, emissions, inputs=None):
        """Compute log marginal likelihood of observations."""
        filtered_posterior = lgssm_filter(self._make_inference_args(params), emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, params, emissions, inputs=None):
        """Compute filtering distribution."""
        return lgssm_filter(self._make_inference_args(params), emissions, inputs)

    def smoother(self, params, emissions, inputs=None):
        """Compute smoothing distribution."""
        return lgssm_smoother(self._make_inference_args(params), emissions, inputs)

    def posterior_sample(self, params, key, emissions, inputs=None):
        _, sample = lgssm_posterior_sample(key, self._make_inference_args(params), emissions, inputs)
        return sample

    def posterior_predictive(self, params, emissions, inputs=None):
        """Compute marginal posterior predictive for each observation.

        Returns:
            means: (T,D) array of E[Y(t,d) | Y(1:T)]
            stds: (T,D) array std[Y(t,d) | Y(1:T)]
        """
        posterior = self.smoother(params, emissions, inputs)
        H = params['emissions']['weights']
        b = params['emissions']['bias']
        R = params['emissions']['cov']
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + b
        smoothed_emissions_cov = H @ posterior.smoothed_covariances @ H.T + R
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    # Expectation-maximization (EM) code
    def e_step(self, params, emissions, inputs=None):
        """The E-step computes sums of expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(self._make_inference_args(params), emissions, inputs)

        # shorthand
        Ex = posterior.smoothed_means
        Exp = posterior.smoothed_means[:-1]
        Exn = posterior.smoothed_means[1:]
        Vx = posterior.smoothed_covariances
        Vxp = posterior.smoothed_covariances[:-1]
        Vxn = posterior.smoothed_covariances[1:]
        Expxn = posterior.smoothed_cross_covariances

        # Append bias to the inputs
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[:self.state_dim, :self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not self.has_emissions_bias:
            emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

    def m_step(self, params, props, batch_stats):

        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = jnp.linalg.solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], None)

        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], None)

        return dict(
            initial=dict(mean=m, cov=S),
            dynamics=dict(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=dict(weights=H, bias=d, input_weights=D, cov=R)
        )
