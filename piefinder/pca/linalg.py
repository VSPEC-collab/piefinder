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
            u = Y[:,row]
            du = dY[:,row]
            v = Y[:,col]
            dv = dY[:,col]
            logl = get_log_likelihood(v,dv,(u,),(du,))
            mat[row,col] = logl
    return mat

def get_basis_indices(logl_mat:np.ndarray):
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
    i,j = np.argwhere(logl_mat==np.nanmin(logl_mat))[0]
    return i,j

def get_ranked_basis_indices(logl_mat:np.ndarray):
    """
    Sort the indicies of the specta to rank them
    by best basis candidate.
    """
    
    def get_next_lowest_index(_logl_mat:np.ndarray,indicies:Tuple[int,...]):
        """
        Get the next lowest index in the matrix.
        """
        mask = np.zeros_like(_logl_mat)
        for index in indicies:
            mask[index,:] = 1
            mask[:,index] = 1
        for i in indicies:
            for j in indicies:
                mask[i,j] = 0
        mask = mask.astype(bool)
        _logl_mat[~mask] = np.nan
        lowest = np.nanmin(_logl_mat[mask])
        ind_low = np.argwhere(_logl_mat==lowest)[0]
        if ind_low[0] in indicies:
            return ind_low[1]
        else:
            return ind_low[0]
        
    mat_min = np.nanmin(logl_mat)
    lowest_indices = np.argwhere(logl_mat==mat_min)[0]
    ranked = list(lowest_indices)
    n_spectra = logl_mat.shape[0]
    while len(ranked) < n_spectra:
        k = get_next_lowest_index(logl_mat,ranked)
        ranked.append(k)
    return tuple(ranked)
        
        
        

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

def get_sum_logl_from_ranked(Y:np.ndarray,dY:np.ndarray,ranked:Tuple[int,...])->float:
    """
    Given a ranked set of indicies, get the log likelihood that the set of spectra
    can be approximated as a linear combination of the spectra specified by
    the indicies.
    """
    len_ranked = len(ranked)
    x = np.arange(1,len_ranked+1)
    y = []
    for _x in x:
        y.append(get_sum_logl_from_indicies(Y,dY,ranked[:_x]))
    return x,y

def get_basis_coeffs(Y:np.ndarray,indices:Tuple[int,...])->np.ndarray:
    """
    Get the coefficients for the basis
    """
    vs = tuple(Y[:,index] for index in indices)
    # dvs = tuple(dY[:,index] for index in indices)
    n_spectra = Y.shape[1]
    return np.array([get_coeffs(Y[:,i],vs) for i in range(n_spectra)]).T
def reconstruct(Y,dY,indices):
    vs = tuple(Y[:,index] for index in indices)
    dvs = tuple(dY[:,index] for index in indices)
    coeffs = get_basis_coeffs(Y,indices)
    Y_approx = np.array([construct_approx(vs,coeffs[:,i]) for i in range(coeffs.shape[1])]).T
    dY_approx = np.array([construct_approx(dvs,coeffs[:,i]) for i in range(coeffs.shape[1])]).T
    return Y_approx,dY_approx
    
    