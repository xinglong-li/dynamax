# Demo of using UKF to track pendulum angle
# Example taken from Simo Särkkä (2013), “Bayesian Filtering and Smoothing,”
from matplotlib import pyplot as plt

import jax.numpy as jnp

from dynamax.nonlinear_gaussian_ssm.demos.simulations import PendulumSimulation
from dynamax.plotting import plot_nlgssm_pendulum as plot_pendulum
from dynamax.nonlinear_gaussian_ssm.containers import NLGSSMParams
from dynamax.unscented_kalman_filter.inference import unscented_kalman_smoother, UKFHyperParams


def ukf_pendulum():
    # Generate random pendulum data
    pendulum = PendulumSimulation()
    states, obs, time_grid = pendulum.sample()

    # Define parameters for UKF
    ukf_params = NLGSSMParams(
        initial_mean=pendulum.initial_state,
        initial_covariance=jnp.eye(states.shape[-1]) * 0.1,
        dynamics_function=pendulum.dynamics_function,
        dynamics_covariance=pendulum.dynamics_covariance,
        emission_function=pendulum.emission_function,
        emission_covariance=pendulum.emission_covariance,
    )
    ukf_hyperparams = UKFHyperParams()

    # Run extended Kalman smoother
    ukf_posterior = unscented_kalman_smoother(ukf_params, obs, ukf_hyperparams)

    return states, obs, time_grid, ukf_posterior


def plot_ukf_pendulum(states, obs, grid, ukf_posterior):
    dict_figures = {}
    dict_figures["ukf_pendulum_data"] = plot_pendulum(grid, states[:, 0], obs)
    dict_figures["ukf_pendulum_filtered"] = plot_pendulum(
        grid, states[:, 0], obs, x_est=ukf_posterior.filtered_means[:, 0], est_type="UKF"
    )
    dict_figures["ukf_pendulum_smoothed"] = plot_pendulum(
        grid, states[:, 0], obs, x_est=ukf_posterior.smoothed_means[:, 0], est_type="UKS"
    )
    return dict_figures


def main(test_mode=False):
    figures = plot_ukf_pendulum(*(ukf_pendulum()))
    return figures

if __name__ == "__main__":
    figures = main()
    plt.show()
