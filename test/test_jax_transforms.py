"""
Tests for piefinder.jax.transforms
"""
from jax import numpy as jnp

from piefinder.jax import transforms

def inverse_testing(tr:transforms.BaseTransform, x):
    """
    Generalized test to test the validity of the inverse transform.
    """
    assert jnp.all(tr.inv(tr(x)) == x), f'failed for x = {x}'

def test_base_transform():
    """
    Test BaseTransform
    """
    
    def func(x):
        return x+1
    def inv(x):
        return x-1
    func = transforms.BaseTransform(func, inv)
    assert func(1) == 2
    assert func.inv(2) == 1
    assert func.inv(func(1)) == 1
    x = jnp.array([1, 2, 3])
    inverse_testing(func, x)
    
    