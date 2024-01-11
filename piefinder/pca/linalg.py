"""
Linear Algebra Utilities
========================
"""
from typing import Tuple
import numpy as np




def get_coeffs(w:np.ndarray,vs:Tuple[np.ndarray]):
    """
    Get the coefficients that the vectors ``vs`` can be multiplied
    by to give an approximation to ``w``.
    
    Parameters
    ----------
    w : np.ndarray
        The vector to be approximated.
    vs : Tuple[np.ndarray]
        The vectors to be used in the approximation.
    """
    vs_reshaped = np.hstack([
        v.reshape(-1,1) for v in vs
    ])
    w_reshaped = w.reshape(-1,1)
    return np.linalg.lstsq(vs_reshaped,w_reshaped,rcond=None)[0]

def construct_approx(vs:Tuple[np.ndarray],coeffs:np.ndarray):
    """
    Construct an approximation to ``w`` using the coefficients
    ``coeffs``.
    """
    return np.sum([coeff*v for coeff,v in zip(coeffs,vs)],axis=0)

def get_residual(w:np.ndarray,vs:Tuple[np.ndarray])->np.ndarray:
    """
    Get the residual between ``w`` and the approximation of ``w``
    using the vectors in ``vs``.
    """
    coeffs = get_coeffs(w,vs)
    approx = construct_approx(vs,coeffs)
    return approx - w

def get_residual_err(dw:np.ndarray,coeffs:np.ndarray,dvs:tuple)->np.ndarray:
    """
    Get the total uncertainty on the residual. This is
    sqrt(err_w**2 + err_approx**2).
    """
    e_approx_sq = construct_approx(dvs,coeffs)**2
    e_tot_sq = e_approx_sq + dw**2
    return np.sqrt(e_tot_sq)

def get_log_likelihood(w:np.ndarray,dw:np.ndarray,vs:tuple,dvs:tuple):
    """
    The log likelihood function.
    """
    coeffs = get_coeffs(w,vs)
    res = get_residual(w,vs)
    e_res = get_residual_err(dw,coeffs,dvs)
    log_likelihood = -0.5 * np.log(2*np.pi) - np.log(e_res) - 0.5*(res/e_res)**2
    return np.sum(log_likelihood)