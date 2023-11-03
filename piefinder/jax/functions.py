"""
Time variability functions.
"""

from typing import Callable
import jax.numpy as jnp
from jax import lax

class BaseFunction:
    """
    A base class for other time variability functions.
    """
    def __init__(self, func: Callable):
        self.func = func
    def __call__(self, t: jnp.ndarray):
        return self.func(t)

def add(func1:BaseFunction,func2:BaseFunction)->BaseFunction:
    """
    Add two functions.
    
    Parameters
    ----------
    func1 : BaseFunction
        The first function.
    func2 : BaseFunction
        The second function.
    
    Returns
    -------
    BaseFunction
    """
    def func(t):
        return func1.func(t) + func2.func(t)
    return BaseFunction(func)
def multiply(func1:BaseFunction,k:BaseFunction)->BaseFunction:
    """
    Multiply a function by a constant.
    
    Parameters
    ----------
    func1 : BaseFunction
        The function.
    k : float
        The constant.
    
    Returns
    -------
    BaseFunction
    """
    def func(t):
        return func1.func(t)*k
    return BaseFunction(func)

class Constant(BaseFunction):
    """
    A constant function.
    """
    def __init__(self, k: float):
        def func(_t):
            return k
        super().__init__(func)