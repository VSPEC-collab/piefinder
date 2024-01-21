"""
priors
"""

from typing import Callable

def uniform(
    a: float,
    b: float
) -> Callable:
    """
    Uniform prior
    """
    def func(u):
        if u<0 or u>1:
            raise ValueError('u must be on the interval [0,1)')
        return a + (b-a)*u
    return func