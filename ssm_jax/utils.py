from functools import partial
import jax.numpy as jnp
from jax import jit
from jax import vmap
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
import tensorflow_probability.substrates.jax.bijectors as tfb

# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
PSDToRealBijector = tfb.Chain([
    tfb.Invert(tfb.FillTriangular()),
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
    tfb.Invert(tfb.CholeskyOuterProduct()),
])


@jit
def pad_sequences(observations, valid_lens, pad_val=0):
    """
    Pad ragged sequences to a fixed length.
    Parameters
    ----------
    observations : array(N, seq_len)
        All observation sequences
    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence
    pad_val : int
        Value that the invalid observable events of the observation sequence will be replaced
    Returns
    -------
    * array(n, max_len)
        Ragged dataset
    """

    def pad(seq, len):
        idx = jnp.arange(1, seq.shape[0] + 1)
        return jnp.where(idx <= len, seq, pad_val)

    dataset = vmap(pad, in_axes=(0, 0))(observations, valid_lens), valid_lens
    return dataset


def monotonically_increasing(x, atol=0, rtol=0):
    thresh = atol + rtol*jnp.abs(x[:-1])
    return jnp.all(jnp.diff(x) >= -thresh)


def add_batch_dim(pytree):
    return tree_map(partial(jnp.expand_dims, axis=0), pytree)


def pytree_len(pytree):
    return len(tree_leaves(pytree)[0])


def pytree_sum(pytree, axis=None, keepdims=None, where=None):
    return tree_map(partial(jnp.sum, axis=axis, keepdims=keepdims, where=where), pytree)


def pytree_stack(pytrees):
    _, treedef = tree_flatten(pytrees[0])
    leaves = [tree_leaves(tree) for tree in pytrees]
    return tree_unflatten(treedef, [jnp.stack(vals) for vals in zip(*leaves)])
