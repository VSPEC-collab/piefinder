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
    return np.linalg.lstsq(vs_reshaped,w,rcond=None)[0].T

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
    
    Essentially, how well can we approximate w with the vectors in vs?
    
    Parameters
    ----------
    w : np.ndarray
        The vector to be approximated.
    dw : np.ndarray
        The uncertainty in the vector to be approximated.
    vs : Tuple[np.ndarray]
        The vectors to be used in the approximation.
    dvs : Tuple[np.ndarray]
        The uncertainties in the vectors to be used in the approximation.
    
    Returns
    -------
    float
        The log likelihood.
    """
    coeffs = get_coeffs(w,vs)
    res = get_residual(w,vs)
    e_res = get_residual_err(dw,coeffs,dvs)
    log_likelihood = -0.5 * np.log(2*np.pi) - np.log(e_res) - 0.5*(res/e_res)**2
    return np.sum(log_likelihood)

def get_logl_matrix(Y:np.ndarray,dY:np.ndarray):
    """
    Get the matrix of log likelihoods.
    
    Parameters
    ----------
    Y : np.ndarray
        The array of spectra. Axis 0 is the spectral axis and
        axis 1 is the time axis.
    dY : np.ndarray
        The uncertainty in the array of spectra.
    
    Returns
    -------
    np.ndarray
        The matrix of log likelihoods.
    """
    N_vec = Y.shape[1]
    mat = np.array(np.zeros((N_vec,N_vec)))*np.nan
    indices = np.triu_indices(N_vec)
    for row, col in zip(indices[0],indices[1]):
        if not row==col:
            indices = (row,col)
            logl = get_sum_logl_from_indicies(Y,dY,indices)
            mat[row,col] = logl
    return mat

def get_first_two_basis_indices(logl_mat:np.ndarray):
    """
    Find the indicies of the likelihood matrix that
    can be used as the basis.
    
    Parameters
    ----------
    logl_mat : np.ndarray
        The matrix of log likelihoods.
    
    Returns
    -------
    Tuple
        The indices of the matrix that can be used as the basis.
    """
    i,j = np.argwhere(logl_mat==np.nanmax(logl_mat))[0]
    return i,j

def get_next_best_index(Y:np.ndarray,dY:np.ndarray,prev_indices:Tuple[int,...]):
    """
    Get the next best basis candidate.
    """
    N_vec = Y.shape[1]
    logl_arr = []
    for i in range(N_vec):
        if i not in prev_indices:
            indices = prev_indices + (i,)
            logl = np.sum(get_logl_from_indicies(Y,dY,indices))
            logl_arr.append(logl)
        else:
            logl_arr.append(np.nan)
    return np.nanargmax(logl_arr)


def get_ranked_basis_indices(Y:np.ndarray,dY:np.ndarray):
    """
    Sort the indicies of the specta to rank them
    by best basis candidate.
    """
    i,j = get_first_two_basis_indices(get_logl_matrix(Y,dY))
    indices = (i,j)
    n_spectra = Y.shape[1]
    while len(indices) < n_spectra:
        i = get_next_best_index(Y,dY,indices)
        indices = indices + (i,)
    return indices
    
        
def get_logl_from_indicies(Y:np.ndarray,dY:np.ndarray,indices:Tuple[int,...])->np.ndarray:
    """
    Given a set of indicies, get the log likelihood that each
    spectrum is a linear combination of the other.
    """
    vs = tuple(Y[:,index] for index in indices)
    dvs = tuple(dY[:,index] for index in indices)
    n_spectra = Y.shape[1]
    return [get_log_likelihood(Y[:,i],dY[:,i],vs,dvs) for i in range(n_spectra)]

def get_sum_logl_from_indicies(Y:np.ndarray,dY:np.ndarray,indices:Tuple[int,...])->float:
    """
    Given a set of indicies, get the log likelihood that the set of spectra
    can be approximated as a linear combination of the spectra specified by
    the indicies.
    """
    vs = tuple(Y[:,index] for index in indices)
    dvs = tuple(dY[:,index] for index in indices)
    n_spectra = Y.shape[1]
    return np.sum([get_log_likelihood(Y[:,i],dY[:,i],vs,dvs) for i in range(n_spectra)])

def get_sum_logl_from_ranked(Y:np.ndarray,dY:np.ndarray,ranked:Tuple[int,...])->Tuple[np.ndarray,np.ndarray]:
    """
    Given a ranked set of indicies, get the log likelihood that the set of spectra
    can be approximated as a linear combination of the spectra specified by
    the indicies.
    """
    len_ranked = len(ranked)
    number_of_bases = np.arange(1,len_ranked+1)
    logl = []
    for _x in number_of_bases:
        logl.append(get_sum_logl_from_indicies(Y,dY,ranked[:_x]))
    return number_of_bases,np.array(logl)

def get_aic(number_of_bases:np.ndarray,logl:np.ndarray):
    """
    Compute the Akaike information criterion
    
    Parameters
    ----------
    number_of_bases : np.ndarray
        The number of basis vectors.
    logl : np.ndarray
        The log likelihoods.
    
    Returns
    -------
    np.ndarray
        The AIC for each model
    
    Notes
    -----
    AIC = -2*logl + 2*number_of_bases
    A better model has a lower AIC.
    """
    return -2*logl + 2*number_of_bases
    

def get_basis_coeffs(Y:np.ndarray,indices:Tuple[int,...])->np.ndarray:
    """
    Get the coefficients for the basis
    """
    vs = tuple(Y[:,index] for index in indices)
    # dvs = tuple(dY[:,index] for index in indices)
    n_spectra = Y.shape[1]
    return np.array([get_coeffs(Y[:,i],vs) for i in range(n_spectra)]).T
def reconstruct(Y:np.ndarray,dY:np.ndarray,indices:Tuple[int,...],coeffs:np.ndarray):
    """
    Reconstruct the phase curve given the basis vectors specified
    by the indicies.
    
    Parameters
    ----------
    Y : np.ndarray
        The array of spectra. Axis 0 is the spectral axis and
        axis 1 is the time axis.
    dY : np.ndarray
        The uncertainty in the array of spectra.
    indices : Tuple[int,...]
        The indicies of the basis vectors.
    """
    vs = tuple(Y[:,index] for index in indices)
    dvs = tuple(dY[:,index] for index in indices)
    
    Y_approx = np.array([construct_approx(vs,coeffs[:,i]) for i in range(coeffs.shape[1])]).T
    dY_approx = np.array([construct_approx(dvs,coeffs[:,i]) for i in range(coeffs.shape[1])]).T
    return Y_approx,dY_approx
    
    