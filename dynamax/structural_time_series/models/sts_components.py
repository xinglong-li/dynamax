from abc import ABC, abstractmethod
from collections import OrderedDict
from dynamax.distributions import InverseWishart as IW
from dynamax.utils import PSDToRealBijector
from dynamax.parameters import ParameterProperties as Prop
from jax import jit
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN)
import tensorflow_probability.substrates.jax.bijectors as tfb


RealToPSD = tfb.Invert(PSDToRealBijector)


#########################
#  Abstract Components  #
#########################

class STSComponent(ABC):
    """Meta class of latent component of structural time series (STS) models.

    A latent component of the STS model has following attributes:

    name (string): name of the latend component.
    dim_obs (int): dimension of the observation in each step of the observed time series.
    initial_distribution (MVN): an instance of MultivariateNormalFullCovariance,
        specifies the distribution for the inital state.
    params (OrderedDict): parameters of the component need to be learned in model fitting.
    param_props (OrderedDict): properties of each item in 'params'.
        Each item is an instance of ParameterProperties, which specifies constrainer
        of the parameter and whether the parameter is trainable.
    priors (OrderedDict): prior distributions for each item in 'params'.
    """

    def __init__(self, name, dim_obs=1):
        self.name = name
        self.dim_obs = dim_obs
        self.initial_distribution = None

        self.param_props = OrderedDict()
        self.priors = OrderedDict()
        self.params = OrderedDict()

    @abstractmethod
    def initialize_params(self, obs_initial, obs_scale):
        """Initialize parameters in self.params given the scale of the observed time series.

        Args:
            obs_initial (self.dim_obs,): the first observation in the observed time series.
            obs_scale (self.dim_obs,): scale of the observed time series.

        Returns:
            No returns. Change self.params and self.initial_distributions directly.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trans_mat(self, params, t):
        """Compute the transition matrix at step t of the latent dynamics.

        Args:
            params (OrderedDict): parameters based on which the transition matrix
                is to be evalueated. Has the same tree structure with self.params.
            t (int): time steps

        Returns:
            trans_mat (dim_of_state, dim_of_state): transition matrix at step t
        """
        raise NotImplementedError

    @abstractmethod
    def get_trans_cov(self, params, t):
        """Nonsingular covariance matrix"""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_mat(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_select_mat(self):
        """Select matrix that makes the covariance matrix singular"""
        raise NotImplementedError


class STSRegression(ABC):
    """Meta class of regression component of structural time series (STS) models.

    The regression component has the same dimension as the observed time series.
    This component will appear in the observation (emission) model in the state space model
    form of the STS, since it does not have the random term, in contrast to latent components.

    A regression component of the STS model has following attributes:

    name (string): name of the regression component.
    dim_obs (int): dimension of the observation in each step of the observed time series.
        This is also the output of the regression model.
    params (OrderedDict): parameters of the function need to be learned in model fitting.
    param_props (OrderedDict): properties of each item in 'params'.
        Each item is an instance of ParameterProperties, which specifies constrainer
        of the parameter and whether the parameter is trainable.
    priors (OrderedDict): prior distributions for each item in 'params'.
    """

    def __init__(self, name, dim_obs=1):
        self.name = name
        self.dim_obs = dim_obs

        self.params = OrderedDict()
        self.param_props = OrderedDict()
        self.priors = OrderedDict()

    @abstractmethod
    def initialize(self, covariates, obs_time_series):
        """Initialize parameters of the regression function by minimising the least square error,
        given the series of covariates and the observed time series.

        Args:
            covariates (len(obs_time_series), dim_covariates): series of inputs
                of the regression function.
            obs_time_series: observed time series.
        Returns:
            No returns. Change self.params and self.initial_distributions directly.
        """
        raise NotImplementedError

    @abstractmethod
    def fitted_values(self, params, covariates):
        """Compute the fitted value of the regression model at one time step
            given parameters and covariates.

        Args:
            params (OrderedDict): parameters based on which the regression model
                is to be evalueated. Has the same tree structure with self.params.
            covariates (dim_cov, ): covariates at which the regression model is to be evaluated

        Raises:
            fitted (dim_obs,): the fitted value of the regression model.
        """
        raise NotImplementedError


#########################
#  Concrete Components  #
#########################


class LocalLinearTrend(STSComponent):
    """The local linear trend component of the structual time series (STS) model

    The latent state is [level, slope], having dimension 2 * dim_obs. The dynamics is

        level[t+1] = level[t] + slope[t] + N(0, cov_level)
        slope[t+1] = slope[t] + N(0, cov_slope)

    In the case dim_obs = 1:

        trans_mat = | 1, 1 |    obs_mat = [ 1, 0 ]
                    | 0, 1 |,
    """

    def __init__(self, dim_obs=1, name='local_linear_trend'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.initial_distribution = MVN(jnp.zeros(2*dim_obs), jnp.eye(2*dim_obs))

        self.param_props['cov_level'] = Prop(trainable=True, constrainer=RealToPSD)
        self.priors['cov_level'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['cov_level'] = self.priors['cov_level'].mode()

        self.param_props['cov_slope'] = Prop(trainable=True, constrainer=RealToPSD)
        self.priors['cov_slope'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['cov_slope'] = self.priors['cov_slope'].mode()

        # The local linear trend component has a fixed transition matrix.
        self._tran_mat = jnp.kron(jnp.array([[1, 1], [0, 1]]), jnp.eye(dim_obs))

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.array([1, 0]), jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.eye(2*dim_obs)

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.tile(obs_initial, 2)
        initial_cov = jnp.kron(jnp.eye(2), jnp.diag(obs_scale))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.priors['cov_level'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['cov_level'] = self.priors['cov_level'].mode()
        self.priors['cov_slope'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['cov_slope'] = self.priors['cov_slope'].mode()

    def get_trans_mat(self, params, t):
        return self._trans_mat

    @jit
    def get_trans_cov(self, params, t):
        _shape = params['cov_level'].shape
        return jnp.block([[params['cov_level'], jnp.zeros(_shape)],
                          [jnp.zeros(_shape), params['cov_slope']]])

    @property
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class SeasonalDummy(STSComponent):
    """The (dummy) seasonal component of the structual time series (STS) model

    Since at any step t the seasonal effect has following constraint

        sum_{j=1}^{num_seasons} s_{t-j} = 0,

    the seasonal effect (random) of next time step takes the form:

        s_{t+1} = - sum_{j=1}^{num_seasons-1} s_{t+1-j} + N(0, drift_cov)

    and the last term is the stochastic noise of the seasonal effect following
    a normal distribution with mean zero(s) and covariance drift_cov.
    So the latent state corresponding to the seasonal component has dimension
    (num_seasons - 1) * dim_obs

    If dim_obs = 1, and suppose num_seasons = 4

                    | -1, -1, -1 |
        trans_mat = |  1,  0   0 |    obs_mat = [ 1, 0, 0 ]
                    |  0,  1,  0 |,

    Args (in addition to name and dim_obs):
        num_seasons (int): number of seasons.
        num_steps_per_season: consecutive steps that each seasonal effect does not change.
            For example, if a STS model has a weekly seasonal effect but the data is measured
            daily, then num_steps_per_season = 7;
            and if a STS model has a daily seasonal effect but the data is measured hourly,
            then num_steps_per_season = 24.
    """

    def __init__(self, num_seasons, num_steps_per_season=1, dim_obs=1, name='seasonal_dummy'):
        super().__init__(name=name, dim_obs=dim_obs)

        self.num_seasons = num_seasons
        self.steps_per_season = num_steps_per_season

        _c = self.num_seasons - 1
        self.initial_distribution = MVN(jnp.zeros(_c*dim_obs), jnp.eye(_c*dim_obs))

        self.param_props['drift_cov'] = Prop(trainable=True, constrainer=RealToPSD)
        self.priors['drift_cov'] = IW(df=dim_obs, scale=jnp.eye(dim_obs))
        self.params['drift_cov'] = self.priors['drift_cov'].mode()

        # The seasonal component has a fixed transition matrix.
        self._trans_mat = jnp.kron(jnp.block([[-jnp.ones(_c)],
                                              [jnp.eye(_c-1), jnp.zeros((_c-1, 1))]]),
                                   jnp.eye(dim_obs))

        # Fixed observation matrix.
        self._obs_mat = jnp.kron(jnp.eye(_c)[0], jnp.eye(dim_obs))

        # Covariance selection matrix.
        self._cov_select_mat = jnp.kron(jnp.eye(_c)[:, 0], jnp.eye(dim_obs))

    def initialize_params(self, obs_initial, obs_scale):
        # Initialize the distribution of the initial state.
        dim_obs = len(obs_initial)
        initial_mean = jnp.zeros((self.num_seasons-1) * dim_obs)
        initial_cov = jnp.kron(jnp.eye(self.num_seasons-1), jnp.diag(obs_scale))
        self.initial_distribution = MVN(initial_mean, initial_cov)

        # Initialize parameters.
        self.priors['drift_cov'] = IW(df=dim_obs, scale=1e-3*obs_scale**2*jnp.eye(dim_obs))
        self.params['drift_cov'] = self.priors['drift_cov'].mode()

    @jit
    def get_trans_mat(self, params, t):
        update = (t % self.steps_per_season == 0)
        if update:
            return self._trans_mat
        else:
            return jnp.eye((self.num_seasons-1) * self.dim_obs)

    @jit
    def get_trans_cov(self, params, t):
        update = (t % self.steps_per_season == 0)
        if update:
            return params['drift_cov']
        else:
            return jnp.zeros((self.dim_obs, self.dim_obs))

    @jit
    def obs_mat(self):
        return self._obs_mat

    @property
    def cov_select_mat(self):
        return self._cov_select_mat


class SeasonalTrig(STSComponent):
    """The trigonometric seasonal component of the structual time series (STS) model
    (Current formulation only support 1-d observation case)

    \gamma_t = \sum_{j=1}^{s/2} \gamma_{jt}
    where
    \gamma_{j, t+1} = \gamma_{jt} cos(\lambda_j) + \gamma*_{jt} sin(\lambda_j) + w_{jt}
    \gamma*_{j, t+1} = -\gamma_{jt} sin(\lambda_j) + \gamma*_{jt} cos(\lambda_j) + w*_{jt}
    for
    j = 1, ..., [s/2], with 's' being the number of seasons

    Args (in addition to name and dim_obs):
        num_seasons (int): number of seasons.
        num_steps_per_season: consecutive steps that each seasonal effect does not change.
            For example, if a STS model has a weekly seasonal effect but the data is measured
            daily, then num_steps_per_season = 7;
            and if a STS model has a daily seasonal effect but the data is measured hourly,
            then num_steps_per_season = 24.
    """

    def __init__(self, num_seasons, num_steps_per_season=1, dim_obs=1, name='seasonal_trig'):
        super().__init__(name=name, dim_obs=dim_obs)

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
    def get_trans_cov(self, params, t):
        return jnp.kron(jnp.eye(self.num_seasons-1), params['drift_cov'])

    @jit
    def obs_mat(self):
        num_pairs = int(jnp.floor(self.num_seasons/2.))
        matrix = jnp.tile(
            jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))]), num_pairs)
        if self.num_seasons % 2 == 0:
            matrix = matrix[:-self.dim_obs, :]
        return matrix

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

        super().__init__(name=name, dim_obs=dim_obs)

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
    def get_trans_mat(self, params, t):
        damp = jnp.diag(self.params['damp'])
        cos_fr = jnp.diag(jnp.cos(self.params['frequency']))
        sin_fr = jnp.diag(jnp.sin(self.params['frequency']))
        return jnp.block([[damp*cos_fr, damp*sin_fr],
                          [-damp*sin_fr, damp*cos_fr]])

    @jit
    def get_trans_cov(self):
        Q = self.params['cov']
        return jnp.block([[Q, jnp.zeros_like(Q)],
                          [jnp.zeros_like(Q), Q]])

    @jit
    def obs_mat(self):
        return jnp.block([jnp.eye(self.dim_obs), jnp.zeros(self.dim_obs, self.dim_obs)])

    @property
    def cov_select_mat(self):
        raise NotImplementedError


class LinearRegression(STSRegression):
    """The linear regression component of the structural time series model.

    Args:
        STSComponent (_type_): _description_
    """
    def __init__(self, dim_covariates, add_bias=True, dim_obs=1, name='linear_regression'):

        self.add_bias = add_bias

        dim_inputs = dim_covariates + 1 if add_bias else dim_covariates

        self.params['weights'] = jnp.zeros((dim_inputs, dim_obs))
        self.param_props['weights'] = None
        self.priors['weights'] = None

    def initialize(self, covariates, obs_time_series):
        if self.add_bias:
            inputs = jnp.concatenate((covariates, jnp.ones(covariates.shape[0], 1)), axis=0)
        W = jnp.solve(inputs.T @ inputs, inputs.T @ observed_time_series)
        self.params['weights'] = W

    def fitted_values(self, params, covariates):
        if self.add_bias:
            return params['weights'] @ jnp.concatenate((covariates, jnp.ones(covariates.shape[0], 1)))
        else:
            return params['weights'] @ covariates


class Autoregressive(STSComponent):
    def __init__(self, p, dim_obs=1, name='ar'):
        super().__init__(name=name, dim_obs=dim_obs)

        # 
        self.initial_distribution = None
        self.params = OrderedDict()
        self.params['coef'] = None
        self.params['noise_cov'] = None
        self.param_props = OrderedDict()
        self.priors = OrderedDict()

        self._obs_mat = jnp.block(
            [jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, (self.num_seasons-2)*self.dim_obs))])

    def initialize_params(self, obs_scale):
        return

    @jit
    def get_trans_mat(self, cur_params, t):
        phi = cur_params['coef']
        order = len(phi)
        if order > 1:
            m = jnp.block([phi[:, None], jnp.vstack((jnp.eye(order-1), jnp.zeros((order-1, 1))))])
        else:
            m = phi[None, :]
        return jnp.kron(m, jnp.eye(self.dim_obs))

    @jit
    def get_trans_cov(self, cur_params, t):
        return self.params['noise_cov']

    @property
    def cov_select_mat(self):
        raise NotImplementedError

    @jit
    def obs_mat(self):
        return self._obs_mat