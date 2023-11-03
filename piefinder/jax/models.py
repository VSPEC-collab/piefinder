"""
Models to use for lightcurve fitting.
"""

from jax import numpy as jnp
from typing import Callable

from GridPolator.grid import GridSpectra

class GridModel:
    def __init__(self, spec: GridSpectra, func: Callable):
        self.spec = spec
        self.func = func

    def evaluate(
        self,
        wl: jnp.ndarray,
        time: jnp.ndarray,
        *args
    ):
        """
        Evaluate the model at the given wavelengths and times.
        
        Parameters
        ----------
        wl : jnp.ndarray
            The wavelengths at which to evaluate the model.
        time : jnp.ndarray
            The times at which to evaluate the model.
        *args
            The other parameters of the model, such as Teff.
        """
        flux: jnp.ndarray = self.func(time).reshape(1, -1) \
            *self.spec.evaluate(wl,*args).reshape(-1, 1)
        if flux.shape != (len(wl), len(time)):
            raise ValueError('Wrong shape!')
        return flux