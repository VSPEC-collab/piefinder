"""
A simple grid retrieval
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


from piefinder.grid.grid import Grid, Parameter
from piefinder.grid import priors

N = 10

params = (
    Parameter('A', np.linspace(0,10,11), priors.uniform(0,10)),
    Parameter('B', np.linspace(0,10,11), priors.uniform(0,10))
)

def model(a,b):
    time = np.linspace(0,10,N)
    wl = np.linspace(0,10,N)
    tt,ww = np.meshgrid(time,wl)
    return a*np.exp(-((tt-5.0)/5.0)**2) * np.exp(-((ww-b)/5.0)**2)
def error(a,b):
    return model(a,b)*0.05

def get_interp():
    points = [param.values for param in params]
    x, y = points
    xx, yy = np.meshgrid(*points)
    points = np.c_[xx.ravel(), yy.ravel()]
    values = np.zeros(shape=(len(x),len(y),N,N))
    for i, _x in enumerate(x):
        for j, _y in enumerate(y):
            values[i,j,:,:] = model(_x,_y)    
    return RegularGridInterpolator((x,y), values)

INTERP = get_interp()


def gridmodel(a,b):
    return INTERP([a,b])

grid = Grid(
    params,
    gridmodel
)

logl = grid.loglike(model(1,5),error(1,5),1,5)

ATRUE = 1
BTRUE = 5

def loglike(x:np.ndarray):
    return grid.loglike(model(ATRUE,BTRUE),error(ATRUE,BTRUE),x[0],x[1])

def ptform(u:np.ndarray):
    return np.array([grid.params[0].prior(u[0]), grid.params[1].prior(u[1])])

sampler = dynesty.NestedSampler(
    loglike,
    ptform,
    ndim=2
)

sampler.run_nested()

results = sampler.results

0