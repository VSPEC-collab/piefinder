"""
A simple example with PCA
=========================

This example shows how to use ``piefinder.pca``
to fit a phase curve.
"""

import numpy as np
from pathlib import Path
from astropy import units as u
import matplotlib.pyplot as plt
import VSPEC
import pypsg

from piefinder import pca

pypsg.docker.set_url_and_run()

FLUX_UNIT = u.Unit('W m-2 um-1')
WL_UNIT = u.um
TIME_UNIT = u.day

#%%
# Create the data
# ---------------
# 
# We will use VSPEC to generate some data.

PATH = Path(__file__).parent / 'pca.yaml'

model = VSPEC.ObservationModel.from_yaml(PATH)
# model.build_planet()
# model.build_spectra()
data = VSPEC.PhaseAnalyzer(model.directories['all_model'])

tot_flux:np.ndarray = data.total.to_value(FLUX_UNIT)
planet_flux:np.ndarray = data.thermal.to_value(FLUX_UNIT)
noise:np.ndarray = data.noise.to_value(FLUX_UNIT)
wl = data.wavelength
time = data.time

fig,ax = plt.subplots(2,1)

ax[0].pcolormesh(time.to_value(TIME_UNIT),wl.to_value(WL_UNIT),tot_flux)
ax[1].pcolormesh(time.to_value(TIME_UNIT),wl.to_value(WL_UNIT),planet_flux)





#%%
# Get a training dataset
# ----------------------
#
# We train on the short wavelength data.

CUTOFF = 3*u.um

ytrain = pca.pipeline.get_training_data(tot_flux,wl,CUTOFF)
eytrain = pca.pipeline.get_training_data(noise,wl,CUTOFF)


#%%
# Rank the spectra
# ----------------
#
# Here we decide which epochs are the most suitable bases.
# We also conpute the AIC for each index.

ranked_indicies = pca.linalg.get_ranked_basis_indices(ytrain,eytrain)

number_of_bases, logl_arr = pca.linalg.get_sum_logl_from_ranked(ytrain,eytrain,ranked_indicies)

aic = pca.linalg.get_aic(number_of_bases,logl_arr)

n_bases = number_of_bases[np.argmin(aic)]

indices_to_use = ranked_indicies[:n_bases]

#%%
# Train the model
# ---------------
#

coeffs = pca.linalg.get_basis_coeffs(ytrain,indices_to_use)

#%%
# Reconstruct
# -----------

yapprox, eyapprox = pca.linalg.reconstruct(tot_flux,noise,indices_to_use,coeffs)

#%%
# Get the residual
# ----------------

residual = yapprox - tot_flux

plt.pcolormesh(time.to_value(TIME_UNIT),wl.to_value(WL_UNIT),residual)

0









