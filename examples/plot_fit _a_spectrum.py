"""
Fit a spectrum
--------------

This example shows how to fit a spectrum using piefinder.

"""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from jax import numpy as jnp, jacobian, jit

from GridPolator.grid import GridSpectra

from piefinder.jax.models import GridModel
from piefinder.jax import functions
from piefinder.jax import transforms
from piefinder.jax.fitting import levenberg_marquardt

rng = np.random.default_rng(42)

w1 = 1*u.um
w2 = 5*u.um
res = 200
teffs:u.Quantity = np.arange(27,34)*100*u.K
times:u.Quantity = np.linspace(-10,10,100)*u.day
wl:u.Quantity = np.linspace(1,5,40)*u.um

#%%
# Get the data
# ------------
spec = GridSpectra.from_vspec(w1,w2,res,teffs)
func = functions.Fourier(20*u.day,0.1,0.2,-0.1)
hot_model = GridModel(
    spec=spec,
    func = func
)
hot_teff = 3200*u.K
cool_model = GridModel(
    spec=spec,
    func = functions.add(
        functions.Constant(1),
        functions.multiply(func,-1)
    )
)
cool_teff = 2800*u.K

data = hot_model.evaluate(wl.value,times,hot_teff) + cool_model.evaluate(wl.value,times,cool_teff)
err = data*0.01
noise = rng.normal(0,err)
data = data+noise


fig,ax = plt.subplots(2,1)

ax[0].pcolormesh(times.value,wl.value,data)
ax[0].set_xlabel(f'time [{times.unit:latex}]')
ax[0].set_ylabel(f'wavelength [{wl.unit:latex}]')

ax[1].plot(times, data[0,:])
ax[1].set_xlabel(f'time [{times.unit:latex}]')
ax[1].set_ylabel(f'flux')

#%%
# Set up the model
# ----------------
#
# Remember that JAX is very particular about types.
# You will get an error if you type `1` instead of `1.`

guess = jnp.array([
    3100., # teff1
    3000., # teff2
    1.,  # scale
    1.,  # coeff1
    1.,  # coeff2
])
tforms = [
    transforms.Log10(),
    transforms.Log10(),
    transforms.Sigmoid(),
    transforms.Sigmoid(),
    transforms.Sigmoid()
]

# @jit
def residuals(
    params:jnp.ndarray,
    w:jnp.ndarray,
    t:jnp.ndarray,
    dat:jnp.ndarray,
    error:jnp.ndarray
):
    inv_params = [tr.inv(p) for p, tr in zip(params,tforms)]
    teff1 = inv_params[0]
    teff2 = inv_params[1]
    scale = inv_params[2]
    coeff1 = inv_params[3]
    coeff2 = inv_params[4]
    model1 = GridModel(
        spec=spec,
        func = functions.multiply(
            functions.Polynomial(coeff1,coeff2),
            scale
        )
    )
    model2 = GridModel(
        spec=spec,
        func = functions.multiply(
            functions.add(
                functions.Constant(1),
                functions.multiply(functions.Polynomial(coeff1,coeff2),-1)),
            scale
        )
    )
    ypred = jnp.array(model1.evaluate(w,t,teff1) + model2.evaluate(w,t,teff2))
    residual = (ypred - dat)/(jnp.maximum(dat,error))
    return residual.reshape(-1,)

@jit
def jac(
    params:jnp.ndarray,
    w:jnp.ndarray,
    t:jnp.ndarray,
    dat:jnp.ndarray,
    error:jnp.ndarray
):
    jac_fun = jacobian(residuals,argnums=0)
    return jac_fun(params,w,t,dat,error)
params = jnp.array([tr(p) for p,tr in zip(guess,tforms)])
result = levenberg_marquardt(residuals,jac, params, (wl,times,data,err),tol=1e-8)

ret_params = jnp.array([tr.inv(p) for p,tr in zip(result.params,tforms)])
print(f'Found teff1 = {ret_params[0]}')
print(f'Found teff2 = {ret_params[1]}')
print(f'Found scale = {ret_params[2]}')
print(f'Found coeff1 = {ret_params[3]}')
print(f'Found coeff2 = {ret_params[4]}')

final_res = residuals(result.params,wl,times,data,err).reshape(data.shape)

fig,ax = plt.subplots(1,1)

im = ax.pcolormesh(times.value,wl.value,final_res/data*1e6)
ax.set_xlabel(f'time [{times.unit:latex}]')
ax.set_ylabel(f'wavelength [{wl.unit:latex}]')

fig.colorbar(im,ax=ax,label = 'residual (ppm)')
0

    
    

