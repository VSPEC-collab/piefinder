"""
Tests for piefinder.jax.functions
"""
from jax import numpy as jnp

from piefinder.jax.functions import BaseFunction, add, multiply

from piefinder.jax import functions

def test_basefunction():
    """
    Test the BaseFunction class
    """
    def func(t):
        return t
    func = BaseFunction(func)
    assert func(1) == 1

def test_add():
    """
    Test the add function
    """
    def func1(t):
        return t
    def func2(t):
        return t
    func1 = BaseFunction(func1)
    func2 = BaseFunction(func2)
    func = add(func1, func2)
    assert func(1) == 2

def test_multiply():
    """
    Test the multiply function
    """
    def func1(t):
        return t
    k = 2
    func1 = BaseFunction(func1)
    func = multiply(func1, k)
    assert func(1) == 2

def test_constant():
    """
    Test the constant function
    """
    k = 2
    func = functions.Constant(k)
    assert func(1) == 2

def test_fourier():
    """
    Test the Fourier function
    """
    lam = 2
    args = [1, 2]
    # = 1*cos(2*pi*0*t/lam)
    # + 2*sin(2*pi*1*t/lam)
    func = functions.Fourier(lam, *args)
    t = 0
    expected = 1
    assert func(t) == expected, f'got {func(t)}, expected {expected}'
    t = 0.5
    expected = 2
    assert func(t) == expected, f'got {func(t)}, expected {expected}'

def test_polynomial():
    """
    Test the Polynomial function
    """
    args = [1, 2]
    # = 1 + 2*t
    func = functions.Polynomial(*args)
    t = 0
    expected = 1
    assert func(t) == expected, f'got {func(t)}, expected {expected}'
    t = 0.5
    expected = 2
    assert func(t) == expected, f'got {func(t)}, expected {expected}'

def test_flare():
    """
    Test the Flare function
    """
    tpeak = 0
    fwhm = 1
    amp = 2
    func = functions.Flare(tpeak, fwhm, amp)
    t = 0
    expected = 2
    assert func(t) == expected, f'got {func(t)}, expected {expected}'
    t = 100
    expected = 0
    assert func(t) == expected, f'got {func(t)}, expected {expected}'
    t = -100
    expected = 0
    assert func(t) == expected, f'got {func(t)}, expected {expected}'
    times = jnp.linspace(-10, 10, 100)
    lc = func(times)
    assert lc.shape == (100,)
    before = times<0
    after = times>0
    assert jnp.all(jnp.diff(lc[before]) >= 0)
    assert jnp.all(jnp.diff(lc[after]) <= 0)
    assert ~jnp.any(lc>amp), 'Flare should not be above amplitude'
    assert ~jnp.any(lc<0), 'Flare should not be negative'