"""
Tests for piefinder.jax.models
"""

from jax import numpy as jnp
from astropy import units as u

from GridPolator.grid import GridSpectra

from piefinder.jax.models import GridModel

def test_gridmodel():
    """
    Test the GridModel class
    """
    
    w1 = 1*u.um
    w2 = 2*u.um
    resolving_power = 100
    teffs = [2900,3000,3100]*u.K
    
    def func(t: jnp.ndarray)-> jnp.ndarray:
        omega = 1/100/u.s
        return jnp.cos(omega*t)
    
    model = GridModel(
        spec=GridSpectra.from_vspec(w1,w2,resolving_power,teffs),
        func=func
    )
    new_wl = jnp.linspace(1,2,100)*u.um
    new_time = jnp.linspace(0,1,100)*u.s
    flux = model.evaluate(new_wl, new_time,3050*u.K)
    assert flux.shape == (100, 100)