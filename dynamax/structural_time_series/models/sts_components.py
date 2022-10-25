from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax import vmap, jit
from dynamax.distributions import InverseWishart as IW
from dynamax.distributions import MatrixNormalPrecision as MN
from dynamax.structural_time_series.models.structural_time_series_ssm import GaussianSSM, PoissonSSM
import optax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


def _set_prior(input_prior, default_prior):
    return input_prior if input_prior is not None else default_prior


class STSLatentComponent(ABC):

    @property
    @abstractmethod
    def transition_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def transition_covariance(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def transition_covariance_prior(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_state_prior(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_spars_matrix(self):
        raise NotImplementedError


class LocalLinearTrend(STSLatentComponent):
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
                 level_covariance_prior=None,
                 slope_covariance_prior=None,
                 initial_level_prior=None,
                 initial_slope_prior=None,
                 observed_time_series=None,
                 dim_observed_timeseries=1,
                 name='LocalLinearTrend'):
        if observed_time_series is not None:
            _dim = observed_time_series.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
            obs_init = observed_time_series[0].mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.
            obs_init = 0.

        self.component_name = name

        # Initialize the prior using the observed time series if a prior is not specified
        self.level_covariance_prior = _set_prior(
            level_covariance_prior,
            IW(df=self.dim_obs, scale=1e-3*obs_scale**2*jnp.eye(self.dim_obs)))

        self.slope_covariance_prior = _set_prior(
            slope_covariance_prior,
            IW(df=self.dim_obs, scale=1e-3*obs_scale**2*jnp.eye(self.dim_obs)))

        self.initial_level_prior = _set_prior(
            initial_level_prior,
            MVN(loc=obs_init * jnp.ones(self.dim_obs),
                covariance_matrix=obs_scale*jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_level_prior, MVN)

        self.initial_slope_prior = _set_prior(
            initial_slope_prior,
            MVN(loc=jnp.zeros(self.dim_obs), covariance_matrix=jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_slope_prior, MVN)

    @property
    def transition_matrix(self):
        return {self.component_name:
                jnp.block([[jnp.eye(self.dim_obs), jnp.eye(self.dim_obs)],
                           [jnp.zeros((self.dim_obs, self.dim_obs)), jnp.eye(self.dim_obs)]])}

    @property
    def observation_matrix(self):
        return {self.component_name:
                jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))])}

    @property
    def transition_covariance(self):
        return OrderedDict({'local_linear_level': self.level_covariance_prior.mode(),
                            'local_linear_slope': self.slope_covariance_prior.mode()})

    @property
    def transition_covariance_prior(self):
        return OrderedDict({'local_linear_level': self.level_covariance_prior,
                            'local_linear_slope': self.slope_covariance_prior})

    @property
    def initial_state_prior(self):
        return OrderedDict({'local_linear_level': self.initial_level_prior,
                            'local_linear_slope': self.initial_slope_prior})

    @property
    def cov_spars_matrix(self):
        return OrderedDict({'local_linear_level': jnp.eye(self.dim_obs),
                            'local_linear_slope': jnp.eye(self.dim_obs)})


class Autoregressive(STSLatentComponent):
    """Autoregressive component
    """

    def __init__(self,
                 order,
                 coefficients_prior=None,
                 level_scale_prior=None,
                 initial_state_prior=None,
                 observed_time_series=None,
                 dim_observed_timeseries=1,
                 name='Autoregressive'):
        if observed_time_series is not None:
            _dim = observed_time_series.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
            obs_init = observed_time_series[0].mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.
            obs_init = 0.

        self.component_name = name

        # Initialize the prior using the observed time series if a prior is not specified
        self.level_covariance_prior = _set_prior(
            level_covariance_prior,
            IW(df=self.dim_obs, scale=1e-3*obs_scale**2*jnp.eye(self.dim_obs)))

        self.slope_covariance_prior = _set_prior(
            slope_covariance_prior,
            IW(df=self.dim_obs, scale=1e-3*obs_scale**2*jnp.eye(self.dim_obs)))

        self.initial_level_prior = _set_prior(
            initial_level_prior,
            MVN(loc=obs_init * jnp.ones(self.dim_obs),
                covariance_matrix=obs_scale*jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_level_prior, MVN)

        self.initial_slope_prior = _set_prior(
            initial_slope_prior,
            MVN(loc=jnp.zeros(self.dim_obs), covariance_matrix=jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_slope_prior, MVN)

    @property
    def transition_matrix(self):
        return {self.component_name:
                jnp.block([[jnp.eye(self.dim_obs), jnp.eye(self.dim_obs)],
                           [jnp.zeros((self.dim_obs, self.dim_obs)), jnp.eye(self.dim_obs)]])}

    @property
    def observation_matrix(self):
        return {self.component_name:
                jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))])}

    @property
    def transition_covariance(self):
        return OrderedDict({'local_linear_level': self.level_covariance_prior.mode(),
                            'local_linear_slope': self.slope_covariance_prior.mode()})

    @property
    def transition_covariance_prior(self):
        return OrderedDict({'local_linear_level': self.level_covariance_prior,
                            'local_linear_slope': self.slope_covariance_prior})

    @property
    def initial_state_prior(self):
        return OrderedDict({'local_linear_level': self.initial_level_prior,
                            'local_linear_slope': self.initial_slope_prior})

    @property
    def cov_spars_matrix(self):
        return OrderedDict({'local_linear_level': jnp.eye(self.dim_obs),
                            'local_linear_slope': jnp.eye(self.dim_obs)})


class LinearRegression():
    """The static regression component of the structual time series (STS) model

    Args:
        weights_prior: MatrixNormal prior for the weight matrix
        weights_shape: Dimension of the observed time series
        name (str): Name of the component in the STS model
    """

    def __init__(self,
                 weights_shape,
                 weights_prior=None,
                 name='LinearRegression'):
        self.dim_obs, self.dim_inputs = weights_shape
        self.component_name = name

        # Initialize the prior distribution for weights
        if weights_prior is None:
            weights_prior = MN(loc=jnp.zeros(weights_shape),
                               row_covariance=jnp.eye(self.dim_obs),
                               col_precision=jnp.eye(self.dim_inputs))

        self.weights_prior = weights_prior


class Seasonal(STSLatentComponent):
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
                 drift_covariance_prior=None,
                 initial_effect_prior=None,
                 observed_time_series=None,
                 dim_observed_timeseries=1,
                 name='Seasonal'):
        if observed_time_series is not None:
            _dim = observed_time_series.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.

        self.num_seasons = num_seasons
        self.num_steps_per_season = num_steps_per_season
        self.component_name = name

        self.initial_effect_prior = _set_prior(
            initial_effect_prior,
            MVN(loc=jnp.zeros(self.dim_obs),
                covariance_matrix=obs_scale**2*jnp.eye(self.dim_obs)))

        self.drift_covariance_prior = _set_prior(
            drift_covariance_prior,
            IW(df=self.dim_obs, scale=1e-3*obs_scale**2*jnp.eye(self.dim_obs)))

    @property
    def transition_matrix(self):
        # TODO: allow num_steps_per_season > 1 or be a list of integers
        return {self.component_name:
                jnp.block([[jnp.kron(-jnp.ones(self.num_seasons-1), jnp.eye(self.dim_obs))],
                           [jnp.eye((self.num_seasons-2)*self.dim_obs),
                            jnp.zeros(((self.num_seasons-2)*self.dim_obs, self.dim_obs))]])}

    @property
    def observation_matrix(self):
        return {self.component_name:
                jnp.block([jnp.eye(self.dim_obs),
                           jnp.zeros((self.dim_obs, (self.num_seasons-2)*self.dim_obs))])}

    @property
    def transition_covariance(self):
        return {'seasonal': self.drift_covariance_prior.mode()}

    @property
    def transition_covariance_prior(self):
        return {'seasonal': self.drift_covariance_prior}

    @property
    def initial_state_prior(self):
        c = self.num_seasons - 1
        initial_loc = jnp.array([self.initial_effect_prior.mean()]*c).flatten()
        initial_cov = jsp.linalg.block_diag(
            *([self.initial_effect_prior.covariance()]*c))
        initial_pri = MVN(loc=initial_loc, covariance_matrix=initial_cov)
        return {'seasonal': initial_pri}

    @property
    def cov_spars_matrix(self):
        return {'seasonal': jnp.concatenate(
                    (
                        jnp.eye(self.dim_obs),
                        jnp.zeros((self.dim_obs*(self.num_seasons-2), self.dim_obs)),
                    ),
                    axis=0)
                }


class SeasonalTrig(STSLatentComponent):
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
                 drift_covariance_prior=None,
                 initial_effect_prior=None,
                 observed_time_series=None,
                 dim_observed_timeseries=1,
                 name='TrigonometricSeasonal'):
        if observed_time_series is not None:
            _dim = observed_time_series.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.

        self.num_seasons = num_seasons
        self.num_steps_per_season = num_steps_per_season
        self.component_name = name

        self.initial_effect_prior = _set_prior(
            initial_effect_prior,
            MVN(loc=jnp.zeros(self.dim_obs),
                covariance_matrix=obs_scale**2*jnp.eye(self.dim_obs)))

        self.drift_covariance_prior = _set_prior(
            drift_covariance_prior,
            IW(df=self.dim_obs, scale=1e-3*obs_scale**2*jnp.eye(self.dim_obs)))

    @property
    def transition_matrix(self):
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

    @property
    def observation_matrix(self):
        num_pairs = int(jnp.floor(self.num_seasons/2.))
        matrix = jnp.tile(jnp.array([1, 0]), num_pairs)
        if self.num_seasons % 2 == 0:
            matrix = matrix[:-1]
        return {self.component_name: matrix[None, :]}

    @property
    def transition_covariance(self):
        # TODO: This formulation does not force all seasons have same drift variance
        covs = {f'season_{j}': self.drift_covariance_prior.mode() for j in range(self.num_seasons-1)}
        return OrderedDict(covs)

    @property
    def transition_covariance_prior(self):
        cov_priors = {f'season_{j}': self.drift_covariance_prior for j in range(self.num_seasons)}
        return OrderedDict(cov_priors)

    @property
    def initial_state_prior(self):
        c = self.num_seasons - 1
        initial_loc = jnp.array([self.initial_effect_prior.mean()]*c).flatten()
        initial_cov = jsp.linalg.block_diag(
            *([self.initial_effect_prior.covariance()]*c))
        initial_pri = MVN(loc=initial_loc, covariance_matrix=initial_cov)
        return {'trig_seasonal': initial_pri}

    @property
    def cov_spars_matrix(self):
        return {'trig_seasonal': jnp.eye(self.num_seasons-1)}
