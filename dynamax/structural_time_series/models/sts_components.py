from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from jax import jit
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    Deterministic)


class STSComponent(ABC):
    """Meta class of latent component of structural time series (STS) models.

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, name, dim_obs=1, *args, **kwargs):
        self.name = name
        self.dim_obs = dim_obs
        self.initial_distribution = None

        self.params = OrderedDict()
        self.param_props = OrderedDict()
        self.priors = OrderedDict()

    @abstractmethod
    def initialize_params(self, obs_scale):
        raise NotImplementedError

    @abstractmethod
    def get_trans_mat(self, cur_params, t):
        raise NotImplementedError

    @abstractmethod
    def get_obs_mat(self, cur_params, t):
        raise NotImplementedError

    @abstractmethod
    def get_trans_cov(self, cur_params, t):
        """Nonsingular covariance matrix"""
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_select_mat(self):
        """Select matrix that makes the covariance matrix singular"""
        raise NotImplementedError


class LocalLinearTrend(STSComponent):
    """The local linear trend component of the structual time series (STS) model

    level[t+1] = level[t] + slope[t] + N(0, level_covariance)
    slope[t+1] = slope[t] + N(0, slope_covariance)

    The latent state is [level, slope].

    Args:
        level_covariance_prior: A tfd.Distribution instance, an InverseWishart prior by default
        slope_covariance_prior: A tfd.Distribution instance, an InverseWishart prior by default
        initial_level_prior: A tfd.Distribution prior for the level part of the initial state,
                             a MultivariateNormal by default
        initial_slope_prior: A tfd.Distribution prior for the slope part of the initial state,
                             a MultivariateNormal by default
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        dim_observed_time_series: dimension of the observed time series
        name (str):               name of the component in the STS model
    """

    def __init__(self,
                 dim_obs=1,
                 name='local_linear_trend'):
        super().__init__()

        self.initial_distribution = None

        self.params['cov_level'] = None
        self.param_props['cov_level'] = None
        self.priors['cov_level'] = None
        
        self.params['cov_slope'] = None
        self.param_props['cov_slope'] = None
        self.priors['cov_slope'] = None

    def initialize_params(self, obs_scale):
        raise NotImplementedError

    @jit
    def get_trans_mat(self, cur_params, t):
        return jnp.block([[jnp.eye(self.dim_obs), jnp.eye(self.dim_obs)],
                          [jnp.zeros((self.dim_obs, self.dim_obs)), jnp.eye(self.dim_obs)]])

    @jit
    def get_obs_mat(self, cur_params, t):
        return jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))])

    @jit
    def get_trans_cov(self, cur_params, t):
        return jnp.block([[cur_params['cov_level'], jnp.zeros((self.dim_obs, self.dim_obs))],
                          [jnp.zeros((self.dim_obs, self.dim_obs)), cur_params['cov_slope']]])

    @property
    def cov_select_mat(self):
        raise NotImplementedError


class Autoregressive(STSComponent):
    def __init__(self, p, dim_obs=1, name='ar'):
        super().__init__()

        self.initial_distribution = None

        self.params = OrderedDict()
        self.param_props = OrderedDict()
        self.priors = OrderedDict()

        self._obs_mat = jnp.block(
            [jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, (self.num_seasons-2)*self.dim_obs))])
    
    def initialize_params(self, obs_scale):
        return
    
    @jit
    def get_trans_mat(self, cur_params, t):
        return

    @jit
    def get_obs_mat(self, cur_params, t):
        return self.obs_mat

    @jit
    def get_trans_cov(self, cur_params, t):
        return

    @property
    def cov_select_mat(self):
        raise NotImplementedError


class SeasonalDummy(STSComponent):
    """The (dummy) seasonal component of the structual time series (STS) model
    Since on average sum_{j=0}^{num_seasons-1}s_{t+1-j} = 0 for any t,
    the seasonal effect (random) for next time step is:

    s_{t+1} = - sum_{j=1}^{num_seasons-1} s_{t+1-j} + N(0, drift_covariance)

    Args:
        num_seasons (int): number of seasons (assuming number of steps per season is 1)
        num_steps_per_season:
        drift_covariance_prior: InverseWishart prior for drift_covariance
        initial_effect_prior: MultivariateNormal prior for initial_effect
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        dim_observed_time_series: dimension of the observed time series
        name (str): name of the component in the STS model
    """

    def __init__(self,
                 num_seasons,
                 num_steps_per_season=1,
                 dim_obs=1,
                 name='seasonal_dummy'):
        super().__init__()

        self.initial_distribution = None
        self.steps_per_season = num_steps_per_season

        self.params['drift_cov'] = None
        self.param_props['drift_cov'] = None
        self.priors['drift_cov'] = None
        
        self._trans_mat = jnp.block(
            [[jnp.kron(-jnp.ones(self.num_seasons-1), jnp.eye(self.dim_obs))],
             [jnp.eye((self.num_seasons-2)*self.dim_obs),
              jnp.zeros(((self.num_seasons-2)*self.dim_obs, self.dim_obs))]])
        self._obs_mat = jnp.block(
            [jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, (self.num_seasons-2)*self.dim_obs))])

    def initialize_params(self, obs_scale):
        raise NotImplementedError

    @jit
    def get_trans_mat(self, cur_params, t):
        update = t % self.steps_per_season == 0
        if update:
            return self.trans_mat
        else:
            return jnp.eye(self.dim_obs)

    @jit
    def get_obs_mat(self, cur_params, t):
        return self.obs_mat

    @jit
    def get_trans_cov(self, cur_params, t):
        update = t % self.steps_per_season == 0
        if update:
            return params['drift_cov']
        else:
            return jnp.zeros((self.dim_obs, self.dim_obs))

    @property
    def cov_select_mat(self):
        raise NotImplementedError


class SeasonalTrig(STSComponent):
    """The trigonometric seasonal component of the structual time series (STS) model
    (Current formulation only support 1-d observation case)

    \gamma_t = \sum_{j=1}^{s/2} \gamma_{jt}
    where
    \gamma_{j, t+1} = \gamma_{jt} cos(\lambda_j) + \gamma*_{jt} sin(\lambda_j) + w_{jt}
    \gamma*_{j, t+1} = -\gamma_{jt} sin(\lambda_j) + \gamma*_{jt} cos(\lambda_j) + w*_{jt}
    for
    j = 1, ..., [s/2], with 's' being the number of seasons

    Args:
        num_seasons (int): number of seasons (assuming number of steps per season is 1)
        num_steps_per_season:
        drift_variance_prior: InverseWishart prior for drift_covariance
        initial_effect_prior: MultivariateNormal prior for initial_effect
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        dim_observed_time_series: dimension of the observed time series
        name (str): name of the component in the STS model
    """

    def __init__(self,
                 num_seasons,
                 num_steps_per_season=1,
                 dim_obs=1,
                 name='seasonal_trig'):
        super().__init__()

        self.initial_distribution = None
        self.num_seasons = num_seasons
        self.num_steps_per_season = num_steps_per_season

        self.params['drift_cov'] = None
        self.params['drift_cov'] = None
        self.priors['drift_cov'] = None

    def initialize_params(self, obs_scale):
        raise NotImplementedError

    @jit
    def get_trans_mat(self, cur_params, t):
        num_pairs = int(jnp.floor(self.num_seasons/2.))
        matrix = jnp.zeros((2*num_pairs, 2*num_pairs))
        for j in 1 + jnp.arange(num_pairs):
            lamb_j = (2*j * jnp.pi) / self.num_seasons
            C = jnp.array([[jnp.cos(lamb_j), jnp.sin(lamb_j)],
                           [-jnp.sin(lamb_j), jnp.cos(lamb_j)]])
            matrix = matrix.at[2*(j-1):2*j, 2*(j-1):2*j].set(C)
        if self.num_seasons % 2 == 0:
            matrix = matrix[:-1, :-1]
        return {self.component_name: matrix}

    @jit
    def get_obs_mat(self, cur_params, t):
        num_pairs = int(jnp.floor(self.num_seasons/2.))
        matrix = jnp.tile(
            jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))]), num_pairs)
        if self.num_seasons % 2 == 0:
            matrix = matrix[:-self.dim_obs, :]
        return matrix

    @jit
    def get_trans_cov(self, params, t):
        return jnp.kron(jnp.eye(self.num_seasons-1), params['drift_cov'])

    @property
    def cov_select_mat(self):
        raise NotImplementedError


class Cycle(STSComponent):
    """The cycle component of the structural time series model

    Args:
        damp (array(dim_ts)): damping factor
        frequency (array(dim_ts)): frequency factor
    """

    def __init__(self, damp=None, frequency=None, covariance=None, dim_obs=1, name='cycle'):
        if damp is not None:
            dim_obs = len(damp)
            assert all(damp > 0.) and all(damp < 1.), "The damping factor shoul be in range (0, 1)."

        super().__init__()

        self.initial_distribution = None

        # Parameters of the component
        self.params['damp'] = damp
        self.param_props['damp'] = None
        self.priors['damp'] = None

        self.params['frequency'] = frequency
        self.param_props['frequency'] = None
        self.priors['frequency'] = None

        self.params['cov'] = covariance
        self.param_props['cov'] = None
        self.priors['cov'] = None
        
    def initialize_params(self, obs_scale):
        raise NotImplementedError

    @jit
    def get_trans_mat(self, cur_params, t):
        damp = jnp.diag(self.params['damp'])
        cos_fr = jnp.diag(jnp.cos(self.params['frequency']))
        sin_fr = jnp.diag(jnp.sin(self.params['frequency']))
        return jnp.block([[damp*cos_fr, damp*sin_fr],
                          [-damp*sin_fr, damp*cos_fr]])

    @jit
    def get_obs_mat(self):
        return jnp.block([jnp.eye(self.dim_obs), jnp.zeros(self.dim_obs, self.dim_obs)])

    @jit
    def get_trans_cov(self):
        Q = self.params['cov']
        return jnp.block([[Q, jnp.zeros_like(Q)],
                          [jnp.zeros_like(Q), Q]])

    @property
    def cov_select_mat(self):
        raise NotImplementedError


class LinearRegression(STSComponent):
    """The linear regression component of the structural time series model.

    Args:
        STSComponent (_type_): _description_
    """
    def __init__(self, covariates, add_bias_term=True, dim_obs=1, name='linear_regression'):
        super().__init__()
        
        if add_bias_term:
            self.inputs = jnp.concatenate((covariates, jnp.ones((len(covariates), 1))), axis=1)
        else:
            self.inputs = covariates
        dim_inputs = self.inputs.shape[-1]
        
        self.initial_distribution = None

        self.params['weights'] = jnp.zeros((dim_inputs, dim_obs))
        self.param_props['weights'] = None
        self.param_props['weights'] = None

    def initialize_params(self):
        W = jnp.solve(c.inputs.T @ c.inputs, c.inputs.T @ observed_time_series)
            c.params['weights'] = W
        residuals = observed_time_series - c.inputs @ W.T
        return residuals

    @jit
    def get_trans_mat(self, cur_params, t):
        return jnp.eye(self.dim_obs)

    @jit
    def get_obs_mat(self, cur_params, t):
        # Set the emission matrix to be the fitted value of the regression model,
        # since we set the latent state for the regression term to be a vector of 1s.
        fitted_value = cur_params['weights'] @ self.inputs[t]
        return jnp.diag(fitted_value)

    @jit
    def get_trans_cov(self, cur_params, t):
        # The regression component does not have a noise term
        return jnp.zeros((self.dim_obs, self.dim_obs))

    @property
    def cov_select_mat(self):
        raise NotImplementedError
