"""
Tests for piefinder.pca.linalg
"""

import numpy as np
import pytest

from piefinder.pca.linalg import (
    get_log_likelihood, get_coeffs, get_logl_matrix,
    get_basis_indices, construct_approx,get_logl_from_indicies,
    get_ranked_basis_indices, get_sum_logl_from_indicies,
    get_sum_logl_from_ranked, get_basis_coeffs, reconstruct
    )

def test_get_coeffs():
    a = np.array([[1,0,0]])
    b = np.array([[0,1,0]])
    c = np.array([[0,0,1]])
    
    w = np.array([1,2,3])
    vs = (a,b,c)
    
    coeffs = get_coeffs(w,vs)
    assert coeffs.ndim == 1, 'coefficients should be 1d'
    assert np.allclose(coeffs,[1,2,3]), f"coefficients should be 1,2,3, got {coeffs}"

def test_log_likelihood():
    a = np.array([1,0,0])
    b = np.array([0,1,0])
    c = np.array([0,0,1])
    
    w = np.array([1,2,3])
    dw = np.array([1e-6,1e-6,1e-6])
    vs = (a,b,c)
    dvs = (dw,dw,dw)
    
    log_likelihood = get_log_likelihood(w,dw,vs,dvs)
    assert log_likelihood > 30
    
    a = np.array([1,0,0])
    b = np.array([0,1,0])
    c = np.array([0,0,1])
    
    w = np.array([1,2,3])
    dw = np.array([1,1,1])
    vs = (a,b,c)
    dvs = (dw,dw,dw)
    log_likelihood = get_log_likelihood(w,dw,vs,dvs)
    assert log_likelihood < 30
    
    a = np.array([1,0,0])
    b = np.array([0,1,0])
    c = np.array([1,1,0])
    
    w = np.array([1,2,3])
    dw = np.array([1e-6,1e-6,1e-6])
    vs = (a,b,c)
    dvs = (dw,dw,dw)
    log_likelihood = get_log_likelihood(w,dw,vs,dvs)
    assert log_likelihood < 0

def test_get_logl_mat():
    """
    Tests for get_logl_mat
    """
    Y = np.array([[1,2,3],[2,4,6],[7,8,9],[10,11,12]]).T
    dy = np.array([[1e-6,1e-6,1e-6],[1e-6,1e-6,1e-6],[1e-6,1e-6,1e-6],[1e-6,1e-6,1e-6]]).T
    mat = get_logl_matrix(Y,dy)
    assert mat.shape == (4,4)

def test_get_basis_indicies():
    """
    Tests for get_basis_indicies
    """
    Y = np.array([[1,0,0],[1,1,0],[-1,2,0],[0,1,0]]).T
    dy = np.array([[1e-6,1e-6,1e-6],[1e-6,1e-6,1e-6],[1e-6,1e-6,1e-6],[1e-6,1e-6,1e-6]]).T
    logl_mat = get_logl_matrix(Y,dy)
    indices = get_basis_indices(logl_mat)
    vs = (Y[:,indices[0]],Y[:,indices[1]])
    for i in range(Y.shape[1]):
        w = Y[:,i]
        coeffs = get_coeffs(w,vs)
        approx = construct_approx(vs,coeffs)
        assert np.allclose(w,approx)
    ranked = get_ranked_basis_indices(logl_mat)
    ind, logl = get_sum_logl_from_ranked(Y,dy,ranked)
    assert logl[-1] > 0
    
    Y = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,1],
        [1,1,1,1],
        [1,1,0,0],
        [1,2,3,4],
        [2,3,2,3]
    ]).T
    dy = np.ones_like(Y)*1e-2
    
    logl_mat = get_logl_matrix(Y,dy)
    ranked = get_ranked_basis_indices(logl_mat)
    ind, logl = get_sum_logl_from_ranked(Y,dy,ranked)
    assert logl[-1] > 0
    coeffs = get_basis_coeffs(Y,ranked[:3])
    assert coeffs.shape[0] == 3
    y_approx, dy_approx = reconstruct(Y,dy,ranked[:3])
    res = Y - y_approx


if __name__ == "__main__":
    pytest.main(args=[__file__])