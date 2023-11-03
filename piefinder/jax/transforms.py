"""
Transforms so variables can be used in
gradient descent more easily.
"""

from typing import Callable

from jax import jit, numpy as jnp

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
        self._func = jit(func)
        self._inv_func = jit(inv_func)
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

class Identity(BaseTransform):
    """
    The identity transform.
    """
    def __init__(self):
        def func(x):
            return x
        def inv(x):
            return x
        super().__init__(func, inv)

class Log10(BaseTransform):
    """
    The log10 transform.
    """
    def __init__(self):
        def func(x):
            return jnp.log10(x)
        def inv(x):
            return jnp.power(10, x)
        super().__init__(func, inv)

class Sigmoid(BaseTransform):
    """
    The sigmoid transform.
    """
    def __init__(self):
        def func(x):
            return 1 / (1 + jnp.exp(-x))
        def inv(x):
            return jnp.log(x / (1 - x))
        super().__init__(func, inv)

class Tanh(BaseTransform):
    """
    The tanh transform.
    
    .. warning::
        This transform is not invertable for large values.
        This is because the floating point precision cannot
        differentiate between tanh(10) and 1.
    """
    def __init__(self):
        def func(x):
            return jnp.tanh(x)
        def inv(x):
            return jnp.arctanh(x)
        super().__init__(func, inv)