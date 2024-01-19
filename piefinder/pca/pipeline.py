"""
PCA Pipeline
============
"""

import numpy as np
from astropy import units as u

from . import linalg

def get_training_data(
    Y:np.ndarray,
    wl:u.Quantity,
    wl_cutoff:u.Quantity,
):
    """
    Get the short wavelength data.
    """
    if len(wl) != Y.shape[0]:
        raise ValueError('The length of the 0th axis of Y must be the same as the length of wl.')
    wl_too_long = wl > wl_cutoff
    return Y[~wl_too_long,:]

def get_reconstruction(Y,eY,wl,wl_cutoff):
    """
    Get the reconstruction.
    """
    Y_train = get_training_data(Y,wl,wl_cutoff)
    eY_train = get_training_data(eY,wl,wl_cutoff)
    
    
    ranked_indices = linalg.get_ranked_basis_indices(Y_train,eY_train)
    number_of_basis, logl = linalg.get_sum_logl_from_ranked(Y_train,eY_train,ranked_indices)
    aic = linalg.get_aic(number_of_basis,logl)
    n_bases = number_of_basis[np.argmin(aic)]
    indices = ranked_indices[:n_bases]
    coeffs = linalg.get_basis_coeffs(Y_train,indices)
    Y_approx, eY_approx = linalg.reconstruct(Y,eY,indices,coeffs)
    return Y_approx, eY_approx