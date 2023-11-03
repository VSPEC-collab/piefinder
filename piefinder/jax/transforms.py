"""
Transforms so variables can be used in
gradient descent more easily.
"""

from typing import Callable

class BaseTransform:
    """
    Base class for transforms.
    
    Parameters
    ----------
    func : Callable
        The function to be transformed.
    inv_func : Callable
        The inverse function.
    """
    def __init__(
        self,
        func: Callable,
        inv_func: Callable
    ):
        self._func = func
        self._inv_func = inv_func
    def __call__(self, x):
        """
        Transform the input.
        
        Parameters
        ----------
        x : jnp.ndarray
            The input.
        
        Returns
        -------
        jnp.ndarray
            The transformed input.
        """
        return self._func(x)
    def inv(self, y):
        """
        Apply the inverse transform.
        
        Parameters
        ----------
        y : jnp.ndarray
            The transformed input.
        
        Returns
        -------
        jnp.ndarray
            The original input.
        """
        return self._inv_func(y)