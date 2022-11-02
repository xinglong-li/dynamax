import jax.numpy as jnp
import jax.random as jr

from dynamax.structural_time_series.models.sts_model import StructuralTimeSeries as STS
from dynamax.structural_time_series.models.sts_components import *
from dynamax.parameters import ParameterProperties as Prop

import tensorflow_probability.substrates.jax as tfp


def latent_comp_infer_test(tfp_comp, comp_from_tfp):
    params = tfp.hmc()
    component_ssm = 
    
    # smoothing
    
    # sampling
    
    # tfp fitting
    
    tfp_dist = decompose_by_component
    mean, cov = 
    
    assert
    
    
    
def latent_comp_forecast_test(component):
    params = None
    component_ssm = as_ssm(component)
    
    forecast_mean = None
    forecast_cov = None
    
    # tfp forecast
    assert 
    

##############################
# Test individual components #
##############################

# Test the LocalLinearTrend comoponent.

local_linear_trend_comp = LocalLinearTrend(dim_obs=1)
local_linear_trend = STS([local_linear_trend_comp], obs_time_series=None,
                         obs_distribution='Gaussian')

tfp_local_linear_trend_comp = tfp.sts.LocalLinearTrend()
tfp_local_linear_trend = tfp.sts.Sum([tfp_local_linear_trend_comp], observed_time_series=None)

def local_linear_from_tfp(comp, tfp_params):
    comp.params['cov_level'] = 
    comp.params['cov_slope'] = 

# Test the Autoregressive component.

autoregress_component = Autoregressive(order=1, dim_obs=1)
autoregress = STS([autoregress_component], obs_time_series=None,
                  obs_distribution='Gaussian')

tfp_autoregress_component = tfp.sts.Autoregressive(order=1)
tfp_autoregress = tfp.sts.Sum([tfp_autoregress_component], observed_time_series=None)

def autoregress_from_tfp(comp):
    comp.params['cov_level'] = 
    comp.params['coef'] = 

# Test the SeasonalDummy component.

season_component = SeasonalDummy(num_seasons=12, num_steps_per_season=1, dim_obs=1)
season = STS([season_component], obs_time_series=None, obs_distribution='Gaussian')

tfp_season_comp = tfp.sts.Seasonal(num_seasons=, num_steps_per_season=)
tfp_season = tfp.sts.Sum([tfp_season_comp], observed_time_series=None)

def season_from_tfp(comp):
    comp

# Test the SeasonalTrig component.

season_trig_comp = SeasonalTrig(num_seasons=12, num_steps_per_season=1, dim_obs=1)
season_trig = STS([season_trig_comp], obs_time_series=None,
                  obs_distribution='Gaussian')

tfp_season_trig_comp = tfp.sts.SmoothSeasonal(period=None, frequency_multipliers=None)
tfp_season_trig = tfp.sts.Sum([tfp_season_trig_comp], observed_time_series=None)


# Test the Cycle component.

cycle_comp = Cycle(dim_obs=1)
cycle = STS([cycle_comp], obs_time_series=None,
            obs_distribution='Gaussian')


# Test the LinearRegression component.

linear_regression_comp = LinearRegression(dim_covariates=1, add_bias=True, dim_obs=1)
linear_regression = STS([linear_regression_comp], obs_time_series=None,
                        obs_distributions=None)

tfp_linear_regression_comp = tfp.sts.LinearRegression(design_matrix=None)
tfp_linear_regression = tfp.sts.Sum([tfp_linear_regression_comp], observed_time_series=None) 
