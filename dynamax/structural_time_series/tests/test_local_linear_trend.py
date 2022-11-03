import jax.numpy as jnp
import jax.random as jr
from jax import lax

from dynamax.structural_time_series.models.sts_model import StructuralTimeSeries as STS
from dynamax.structural_time_series.models.sts_components import *
from dynamax.parameters import ParameterProperties as Prop

import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiag as MVNDiag
)


def test_local_linear_trend_forecast(time_steps=200, key=jr.PRNGKey(0)):

    keys = jr.split(key, 5)
    standard_mvn = MVNDiag(jnp.zeros(1), jnp.ones(1))

    # Generate parameters of the STS component
    level_scale = 5
    slope_scale = 0.5
    initial_level = standard_mvn.sample(seed=keys[0])
    initial_slope = standard_mvn.sample(seed=keys[1])

    obs_noise_scale = 20

    # Generate observed time series using the SSM representation.
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])
    Q = jnp.block([[level_scale, 0],
                   [0, slope_scale]])
    R = obs_noise_scale

    def _step(current_state, key):
        key1, key2 = jr.split(key)
        current_obs = H @ current_state + R * standard_mvn.sample(seed=key1)
        next_state = F @ current_state + Q @ MVNDiag(jnp.zeros(2), jnp.ones(2)).sample(seed=key2)
        return next_state, current_obs

    initial_state = jnp.concatenate((initial_level, initial_slope))
    key_seq = jr.split(keys[2], time_steps)
    _, obs_time_series = lax.scan(_step, initial_state, key_seq)

    # Fit the STS model using tfp module
    tfp_comp = tfp.sts.LocalLinearTrend(observed_time_series=obs_time_series)
    tfp_model = tfp.sts.Sum([tfp_comp], observed_time_series=obs_time_series)

    tfp_samples, _ = tfp.sts.fit_with_hmc(
        num_results=100, model=tfp_model, observed_time_series=obs_time_series, seed=keys[3])

    # Build the dynamax component.
    dynamax_comp = LocalLinearTrend(name='local_linear_trend')
    dynamax_model = STS([dynamax_comp], obs_time_series=obs_time_series)

    # Set the parameters to the parameters learned by the tfp module and fix the parameters.
    dynamax_model.params['local_linear_trend']['cov_level'] = tfp_samples[1]
    dynamax_model.params['local_linear_trend']['cov_level'] = tfp_samples[2]
    dynamax_model.params['obs_model']['cov'] = tfp_samples[0]

    dynamax_model.param_props['local_linear_trend']['cov_level'].trainable = False
    dynamax_model.param_props['local_linear_trend']['cov_level'].trainable = False
    dynamax_model.param_props['obs_model']['cov'].trainble = False

    # Fit and forecast with the tfp module
    tfp.sts.forecast(model, observed_time_series, parameter_samples, num_steps_forecast,
    include_observation_noise=True)
    tfp.sts.decompose_by_component(model, observed_time_series, parameter_samples)

    tfp_posterior_mean = None
    tfp_posterior_cov = None
    dynamax_posterior_mean = None
    dynamax_posterior_cov = None

    # Fit and forecast with dynamax
    tfp_forecast_mean = None
    tfp_forecast_cov = None
    dynamax_forecast_mean = None
    dynamax_forecast_cov = None

    # Compare
    assert jnp.allclose(tfp_posterior_mean, dynamax_posterior_mean, rtol=1e-2)
    assert jnp.allclose(tfp_posterior_cov, dynamax_posterior_cov, rtol=1e-2)
    assert jnp.allclose(tfp_forecast_mean, dynamax_forecast_mean, rtol=1e-2)
    assert jnp.allclose(tfp_forecast_cov, dynamax_forecast_cov, rtol=1e-2)


def test_local_linear_trend_hmc(time_steps=200, key=jr.PRNGKey(0)):

    keys = jr.split(key, 5)
    standard_mvn = MVNDiag(jnp.zeros(1), jnp.ones(1))

    # Generate parameters of the STS component
    level_scale = 5
    slope_scale = 0.5
    initial_level = standard_mvn.sample(seed=keys[0])
    initial_slope = standard_mvn.sample(seed=keys[1])

    obs_noise_scale = 20

    # Generate observed time series using the SSM representation.
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])
    Q = jnp.block([[level_scale, 0],
                   [0, slope_scale]])
    R = obs_noise_scale

    def _step(current_state, key):
        key1, key2 = jr.split(key)
        current_obs = H @ current_state + R * standard_mvn.sample(seed=key1)
        next_state = F @ current_state + Q @ MVNDiag(jnp.zeros(2), jnp.ones(2)).sample(seed=key2)
        return next_state, current_obs

    initial_state = jnp.concatenate((initial_level, initial_slope))
    key_seq = jr.split(keys[2], time_steps)
    _, obs_time_series = lax.scan(_step, initial_state, key_seq)
