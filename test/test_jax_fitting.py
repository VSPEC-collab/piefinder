"""
Test piefinder.jax.fitting
"""

import pytest
from jax import numpy as jnp, jit, jacobian

from numpy.random import default_rng

from piefinder.jax.fitting import levenberg_marquardt, FitResult

def test_levenberg_marquardt():
    """
    Test levenberg_marquardt
    """
    rng = default_rng(seed=10)
    m = 3
    b = 2
    x = jnp.linspace(-10,10,100)
    y = m*x + b + rng.normal(0,0.1,100)
    @jit
    def residuals(p, x, y):
        return y - p[0]*x - p[1]
    @jit
    def jac(p, x, y):
        return jacobian(residuals,argnums=0)(p, x, y)    
    
    result = levenberg_marquardt(residuals,jac, jnp.array([1.,1.]), (x, y),tol=1e-4)
    assert isinstance(result, FitResult)
    assert result.success
    assert result.params[0] == pytest.approx(m, rel=1e-2)
    assert result.params[1] == pytest.approx(b, rel=1e-2)
    
    
if __name__ in '__main__':
    test_levenberg_marquardt()