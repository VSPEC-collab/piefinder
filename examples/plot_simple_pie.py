"""
Simple PIE Example
==================

This is a basic example of how to use ``piefinder.jax``
to find planetary infrared excess.

"""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from typing import List

from jax import numpy as jnp, jacobian, jit

from GridPolator.grid import GridSpectra

from piefinder.jax.models import GridModel
from piefinder.jax import functions
from piefinder.jax import transforms
from piefinder.jax.fitting import levenberg_marquardt

rng = np.random.default_rng(10)

w1 = 1*u.um
w2 = 20*u.um
res = 200
teffs:u.Quantity = np.arange(27,34)*100*u.K
times:u.Quantity = np.linspace(-10,10,100)*u.day
wl:u.Quantity = np.linspace(1,20,40)*u.um
cutoff:u.Quantity = 5*u.um

#%%
# Get the data
# ------------

true_lam = 20*u.day
c1_true = 0.1
c2_true = 0.2
c3_true = -0.1
spec = GridSpectra.from_vspec(w1,w2,res,teffs)
func = functions.Fourier(true_lam,c1_true,c2_true,c3_true)
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
err = data*0.003
noise = rng.normal(0,err)
data = data+noise


fig,ax = plt.subplots(2,1)

im = ax[0].pcolormesh(times.value,wl.value,np.log10(data))
ax[0].set_xlabel(f'time [{times.unit:latex}]')
ax[0].set_ylabel(f'wavelength [{wl.unit:latex}]')
fig.colorbar(im,ax=ax[0],label='Log Flux/[W m-2 s-1 um-1]')

ax[1].plot(times, data[0,:])
ax[1].set_xlabel(f'time [{times.unit:latex}]')
ax[1].set_ylabel('flux')

#%%
# Create a planet phase curve
# --------------------------
#
tt, ww = np.meshgrid(times, wl)
pl_norm = 1e1
pl_time = (np.sin((tt*np.pi/(10*u.day)).to_value(u.dimensionless_unscaled)) + 1.3)/2
pl_wl = np.exp(-((ww-15*u.um)/(6*u.um))**2)
pl = pl_norm*pl_time*pl_wl

fig,ax = plt.subplots(2,1)

im = ax[0].pcolormesh(times.value,wl.value,pl)
ax[0].set_xlabel(f'time [{times.unit:latex}]')
ax[0].set_ylabel(f'wavelength [{wl.unit:latex}]')
fig.colorbar(im,ax=ax[0],label='Flux (W m-2 s-1 um-1)')

im = ax[1].pcolormesh(times.value,wl.value,pl/data*100)
ax[1].set_xlabel(f'time [{times.unit:latex}]')
ax[1].set_ylabel(f'wavelength [{wl.unit:latex}]')
fig.colorbar(im,ax=ax[1],label='Flux (%)')


#%%
# Add the planet to the data
# -------------------------
data = data + pl

fig,ax = plt.subplots(1,1)

im = ax.pcolormesh(times.value,wl.value,np.log10(data))
ax.set_xlabel(f'time [{times.unit:latex}]')
ax.set_ylabel(f'wavelength [{wl.unit:latex}]')
fig.colorbar(im,ax=ax,label='Log Flux/[W m-2 s-1 um-1]')

#%%
# Create a training dataset
# ------------------------
# We will use 5 microns as the cutoff.

data_train = data[wl.value<cutoff.value,:]
err_train = err[wl.value<cutoff.value,:]
wl_train = wl[wl.value<cutoff.value]






#%%
# Set up the model
# ----------------
#
# Remember that JAX is very particular about types.
# You will get an error if you type `1` instead of `1.`

guess = jnp.array([
    3200., # teff1
    2800., # teff2
    1.,  # scale
    1.,  # coeff1
    1.,  # coeff2
    1.,  # coeff3
])
tforms:List[transforms.BaseTransform] = [
    transforms.Log10(),
    transforms.Log10(),
    transforms.Sigmoid(),
    transforms.Sigmoid(),
    transforms.Sigmoid(),
    transforms.Sigmoid()
]

@jit
def get_model(
    params:jnp.ndarray,
    w:jnp.ndarray,
    t:jnp.ndarray,
):
    inv_params = [tr.inv(p) for p, tr in zip(params,tforms)]
    teff1 = inv_params[0]
    teff2 = inv_params[1]
    scale = inv_params[2]
    coeff1 = inv_params[3]
    coeff2 = inv_params[4]
    coeff3 = inv_params[5]
    model1 = GridModel(
        spec=spec,
        func = functions.multiply(
            functions.Fourier(true_lam,coeff1,coeff2,coeff3),
            scale
        )
    )
    model2 = GridModel(
        spec=spec,
        func = functions.multiply(
            functions.add(
                functions.Constant(1),
                functions.multiply(functions.Fourier(true_lam,coeff1,coeff2,coeff3),-1)),
            scale
        )
    )
    ypred = jnp.array(model1.evaluate(w,t,teff1) + model2.evaluate(w,t,teff2))
    return ypred


@jit
def residuals(
    params:jnp.ndarray,
    w:jnp.ndarray,
    t:jnp.ndarray,
    dat:jnp.ndarray,
    error:jnp.ndarray
):
    ypred = get_model(params,w,t)
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
result = levenberg_marquardt(residuals,jac, params, (wl_train,times,data_train,err_train),tol=1e-9)

ret_params = jnp.array([tr.inv(p) for p,tr in zip(result.params,tforms)])
print(f'Found teff1 = {ret_params[0]}')
print(f'Found teff2 = {ret_params[1]}')
print(f'Found scale = {ret_params[2]}')
print(f'Found coeff1 = {ret_params[3]}')
print(f'Found coeff2 = {ret_params[4]}')
print(f'Found coeff3 = {ret_params[5]}')


#%%
# Plot the residuals
# -----------------
# The residuals should contain the planetary infrared excess.

final_model = get_model(result.params,wl,times)

excess =  data - final_model

fig,ax = plt.subplots(1,1)

im = ax.pcolormesh(times.value,wl.value,excess/data*100)
ax.set_xlabel(f'time [{times.unit:latex}]')
ax.set_ylabel(f'wavelength [{wl.unit:latex}]')

_=fig.colorbar(im,ax=ax,label = 'residual (%)')

#%%
# Plot the planet spectrum
# ------------------------

im_to_use = 75
fig,ax = plt.subplots(1,1)
ax.errorbar(wl.value,(excess/data)[:,im_to_use]*100,label='PIE',
            yerr=err[:,im_to_use]/data[:,im_to_use]*100,
            fmt='o')
ax.plot(wl.value,(pl/data)[:,im_to_use]*100,label='true')

ax.set_xlabel(f'wavelength [{wl.unit:latex}]')
ax.set_ylabel('flux %')
ax.legend()

    
    

