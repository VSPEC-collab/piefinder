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

class Fourier(BaseFunction):
    """
    A Fourier function.
    
    Parameters
    ----------
    lam : float
        The longest wavelength to consider.
    *args
        The Fourier coefficients.
    """
    def __init__(self, lam: float, *args: float):
        def func(t):
            sum = jnp.zeros_like(t)
            for i, arg in enumerate(args):
                n = i // 2 + 1  # Compute the Fourier series index (n)
                if i % 2 == 0:
                    sum += arg * jnp.cos(2 * jnp.pi * n * t / lam)  # Cosine term
                else:
                    sum += arg * jnp.sin(2 * jnp.pi * n * t / lam)  # Sine term
            return sum
        super().__init__(func)

class Polynomial(BaseFunction):
    """
    A polynomial function.
    
    Parameters
    ----------
    *args
        The polynomial coefficients.
    """
    def __init__(self, *args: float):
        def func(t):
            sum = jnp.zeros_like(t)
            for i, arg in enumerate(args):
                sum += arg * jnp.power(t, i)
            return sum
        super().__init__(func)

class Flare(BaseFunction):
    """
    A flare function.
    
    Parameters
    ----------
    tpeak : float
        The peak time of the flare.
    fwhm : float
        The full width at half maximum of the flare.
    amp : float
        The amplitude of the flare.
    
    Notes
    -----
    This is a reimplementation of code written
    by Tom Barclay, https://github.com/mrtommyb/xoflares
    """
    def __init__(
        self,
        tpeak: float,
        fwhm: float,
        amp: float
    ):
        def func(t):
            t_half = (t - tpeak) / fwhm
            coeffs = [
                1.0,
                1.941,
                -0.175,
                -2.246,
                -1.125
            ]
            f_rise = lax.bitwise_and(t_half >= -1, t_half <= 0) * (
                coeffs[0] + coeffs[1] * t_half + coeffs[2] * jnp.power(t_half, 2) + coeffs[3] * jnp.power(t_half, 3) + coeffs[4] * jnp.power(t_half, 4)
            )
            coeffs = [
                0.6890,
                -1.600,
                0.3030,
                -0.2783,
            ]
            f_decay = lax.bitwise_and(t_half > 0, t_half <= 20) * (
                coeffs[0] * jnp.exp(coeffs[1] * t_half) + coeffs[2] * jnp.exp(coeffs[3] * t_half)
            )

            return (f_rise + f_decay) * amp
        super().__init__(func)