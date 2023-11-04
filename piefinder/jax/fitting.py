"""
Fitting module
"""

from typing import Callable, Tuple
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from time import time


class FitResult:
    """
    The result of a fit.

    Parameters
    ----------
    success : bool
        Whether the fit converged.
    n_iter : int
        The number of iterations.
    runtime : float
        The runtime of the fit in seconds.
    params : jnp.ndarray
        The fitted parameters.
    param_history : jnp.ndarray
        The history of the parameters.

    Attributes
    ----------
    success : bool
        Whether the fit converged.
    n_iter : int
        The number of iterations.
    runtime : float
        The runtime of the fit in seconds.
    params : jnp.ndarray
        The fitted parameters.
    param_history : jnp.ndarray
        The history of the parameters.

    Methods
    -------
    get_indicies(step=False)
        Get the indicies of the parameters.

    """

    def __init__(self, success: bool, n_iter: int, runtime: float, params: jnp.array, param_history: jnp.array, **kwargs):
        self.success = success
        self.n_iter = n_iter
        self.runtime = runtime
        self.params = params
        self.param_history = param_history

    def get_indicies(self, step=False):
        """
        Get the indicies of the parameters.

        Parameters
        ----------
        step : bool
            Whether to return the indicies of the steps.

        Returns
        -------
        jnp.ndarray
            The indicies of the parameters.
        """
        if step:
            return np.arange(self.n_iter-1)
        return np.arange(self.n_iter)


def levenberg_marquardt(
    objective: Callable,
    jacobian: Callable,
    initial_params: jnp.ndarray,
    args: Tuple,
    max_iterations: int = 100,
    tol: float = 1e-6,
    lambda_init: float = 0.1,
    verbose: int = 1
):
    """
    A Levenberg-Marquardt algorithm for fitting.

    Parameters
    ----------
    objective : Callable
        The objective function.
    jacobian : Callable
        The Jacobian of the objective function.
    initial_params : jnp.ndarray
        The initial parameters.
    args : Any
        The arguments of the objective function.
    max_iterations : int
        The maximum number of iterations.
    tol : float
        The tolerance for convergence.
    lambda_init : float
        The initial lambda value.
    verbose : int
        The verbosity level.

    Returns
    -------
    FitResult
        The result of the fit.
    """
    params = initial_params
    lambda_lm = lambda_init
    start_time = time()
    param_hist = []

    for i in range(max_iterations):
        residuals = objective(params, *args)
        if (i % 20 == 0) and (verbose > 0):
            log_mean_residual = jnp.log10(jnp.mean(jnp.abs(residuals)))
            print(
                f'Starting iteration {i}. Log(mean(|residual|)) = {log_mean_residual:.1f}')

        jac = jacobian(params, *args)
        hess = jnp.dot(jnp.transpose(jac), jac)
        lm_update = jnp.linalg.solve(
            hess + lambda_lm * jnp.eye(len(params)),
            jnp.matmul(jnp.transpose(jac), residuals)
        )

        new_params = params - lm_update

        new_residuals = objective(new_params, *args)
        new_objective_value = jnp.dot(new_residuals, new_residuals)

        # Check convergence criteria
        if jnp.linalg.norm(new_params - params) < tol:
            runtime = time() - start_time
            if verbose > 0:
                print(
                    f'Converged with condition 1 after {i} iterations ({runtime:.2f} s)')
            return FitResult(True, i+1, runtime, params, jnp.array(param_hist).T)
        if new_objective_value < tol:
            runtime = time() - start_time
            if verbose > 0:
                print(
                    f'Converged with condition 2 after {i} iterations ({runtime:.2f} s)')
            return FitResult(True, i+1, runtime, params, jnp.array(param_hist).T)
        # Update parameter values and lambda_lm for next iteration
        if new_objective_value < jnp.dot(residuals, residuals):
            params = new_params
            param_hist.append(params)
            lambda_lm /= 10
        else:
            lambda_lm *= 10
    runtime = time() - start_time
    if verbose > 0:
        print(f'Did not converge after {i} iterations ({runtime:.2f} s)')
    return FitResult(False, i+1, runtime, params, jnp.array(param_hist).T)
