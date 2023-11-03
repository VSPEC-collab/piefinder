"""
Tests for piefinder.jax.transforms
"""
from jax import numpy as jnp, config
import pytest

from piefinder.jax import transforms

config.update("jax_enable_x64", True)


def inverse_testing(tr:transforms.BaseTransform, x):
    """
    Generalized test to test the validity of the inverse transform.
    """
    try:
        assert jnp.all(jnp.isclose(tr.inv(tr(x)), x, rtol=1e-4)), f'failed for x = {x}'
    except AssertionError:
        for elem in x:
            assert jnp.isclose(tr.inv(tr(elem)), elem, rtol=1e-4), f'failed for x = {elem}, {(tr.inv(tr(elem))-elem)/elem*100} % error'

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
    
def test_identity():
    """
    Test Identity
    """
    func = transforms.Identity()
    assert func(1) == 1
    assert func.inv(1) == 1
    x = jnp.array([1, 2, 3])
    inverse_testing(func, x)

def test_log10():
    """
    Test Log10
    """
    func = transforms.Log10()
    assert func(10) == pytest.approx(1, rel=1e-6)
    assert func.inv(1) == pytest.approx(10, rel=1e-6)
    x = jnp.array([0.1 ,1, 10, 100])
    inverse_testing(func, x)
    
def test_sigmoid():
    """
    Test Sigmoid
    """
    func = transforms.Sigmoid()
    assert func(0) == pytest.approx(0.5, rel=1e-6)
    assert func.inv(0.5) == pytest.approx(0, rel=1e-6)
    x = jnp.array([-10, -1, 0, 1, 10])
    inverse_testing(func, x)

def test_tanh():
    """
    Test Tanh
    
    """
    func = transforms.Tanh()
    assert func(0) == pytest.approx(0, rel=1e-6)
    assert func(4) < 1.0
    assert func.inv(0) == pytest.approx(0, rel=1e-6)
    x = jnp.array([-10, -1, 0, 1, 10])
    inverse_testing(func, x)