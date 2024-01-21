"""
Grid retrieval
"""

from typing import Callable, Tuple
import numpy as np

import dynesty


class Parameter:
    """
    A grid axis parameter
    """
    def __init__(
        self,
        name: str,
        values: np.ndarray,
        prior: Callable
    ):
        self.name = name
        self.values = values
        self._prior = prior
    def prior(self,u:float | np.ndarray) -> float | np.ndarray:
        match u:
            case float():
                if u<0 or u>=1:
                    raise ValueError('u must be on the interval [0,1)')
            case np.ndarray():
                if np.any(u<0) or np.any(u>=1):
                    raise ValueError('u must be on the interval [0,1)')
        return self._prior(u)

class Grid:
    """
    A grid of models
    """
    def __init__(
        self,
        params: Tuple[Parameter,...],
        model: Callable
    ):
        self.params = params
        self._model = model
    @property
    def n_params(self):
        return len(self.params)
    def get_model(self, *args):
        return self._model(*args)
    def loglike(self,data,err, *args):
        model = self.get_model(*args)
        res = data-model
        logl = -0.5 * np.log(2*np.pi) - np.log(err) - 0.5*(res/err)**2
        return np.sum(logl)