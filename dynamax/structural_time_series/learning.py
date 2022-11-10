import blackjax
from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map, tree_flatten, tree_leaves
from jaxopt import LBFGS
from dynamax.parameters import to_unconstrained, from_unconstrained, log_det_jac_constrain
from dynamax.utils import ensure_array_has_batch_dim, pytree_slice, pytree_stack


def fit_vi(model,
           initial_params,
           param_props,
           sample_size,
           emissions,
           inputs=None,
           key=jr.PRNGKey(0),
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
    keys = iter(jr.split(key, 3*len(tree_leaves(initial_params))))

    # Make sure the emissions and covariates have batch dimensions
    batch_emissions = ensure_array_has_batch_dim(emissions, model.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, model.inputs_shape)

    curr_unc_params, fixed_params = to_unconstrained(initial_params, param_props)

    # standard nornal samples
    std_samples = tree_map(lambda x: jr.normal(next(keys), (M, *x.shape)), curr_unc_params)

    @jit
    def unnorm_log_pos(_unc_params):
        params = from_unconstrained(_unc_params, fixed_params, param_props)
        log_det_jac = log_det_jac_constrain(_unc_params, fixed_params, param_props)
        log_pri = model.log_prior(params) + log_det_jac
        batch_lls = vmap(partial(model.marginal_log_prob, params))(batch_emissions, batch_inputs)
        lp = log_pri + batch_lls.sum()
        return lp

    @jit
    def neg_elbo(vi_hyper):
        """Evaluate negative ELBO at fixed samples from the approximate distribution q.
        """
        # Turn VI parameters and fixed noises into samples of unconstrained parameters of q.
        unc_params = lambda samp: tree_map(lambda p, s: p[0] + jnp.exp(p[1])*s, vi_hyper, samp)
        log_probs = jnp.array([unnorm_log_pos(unc_params(pytree_slice(std_samples, i)))
                               for i in range(M)]).mean()
        vi_hyper_flat = tree_leaves(vi_hyper)
        q_entropy = jnp.array([p[1].sum() for p in vi_hyper_flat]).sum()
        return -(log_probs + q_entropy)

    # Fit ADVI with LBFGS algorithm
    curr_vi_means = curr_unc_params
    curr_vi_log_sigmas = tree_map(lambda x: jnp.zeros(x.shape), curr_unc_params)
    curr_vi_hyper = tree_map(lambda x, y: jnp.stack((x, y)), curr_vi_means, curr_vi_log_sigmas)
    lbfgs = LBFGS(fun=neg_elbo)
    vi_hyp_fitted, _info = lbfgs.run(curr_vi_hyper)

    # Sample from the learned approximate posterior q
    vi_unc_samples = tree_map(
        lambda p: p[0] + jnp.exp(p[1])*jr.normal(next(keys), (sample_size, *p[0].shape)),
        vi_hyp_fitted)
    _samples = [from_unconstrained(pytree_slice(vi_unc_samples, i), fixed_params, param_props)
                for i in range(sample_size)]
    vi_samples = pytree_stack(_samples)

    return vi_samples


def fit_hmc(model,
            initial_params,
            param_props,
            num_samples,
            emissions,
            inputs=None,
            key=jr.PRNGKey(0),
            warmup_steps=100,
            verbose=True):
    """Sample parameters of the model using HMC.
    """
    # Make sure the emissions and covariates have batch dimensions
    batch_emissions = ensure_array_has_batch_dim(emissions, model.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, model.inputs_shape)

    initial_unc_params, fixed_params = to_unconstrained(initial_params, param_props)

    # The log likelihood that the HMC samples from
    def unnorm_log_pos(_unc_params):
        params = from_unconstrained(_unc_params, fixed_params, param_props)
        log_det_jac = log_det_jac_constrain(_unc_params, fixed_params, param_props)
        log_pri = model.log_prior(params) + log_det_jac
        batch_lls = vmap(partial(model.marginal_log_prob, params))(batch_emissions, batch_inputs)
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
    pbar = progress_bar(range(num_samples)) if verbose else range(num_samples)
    for _ in pbar:
        step_key, key = jr.split(key)
        hmc_state, params = hmc_step(hmc_state, step_key)
        log_probs.append(-hmc_state.potential_energy)
        samples.append(params)

    # Combine the samples into a single pytree
    return pytree_stack(samples), jnp.array(log_probs)
