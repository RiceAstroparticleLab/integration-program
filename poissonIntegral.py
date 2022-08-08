import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit
from math import log, pi
import xarray as xr
import zarr
# from ramanujan_pmf import ramanujan_pmf
@jit(nopython=True)

def r(x):
    """Ramanujan log-gamma approximation in place of log factorial
        in pmf function

        :param x:
        :return: ramanujan log-gamma approximation
    """
    if x == 0:
        return 1
    return x*log(x) - x + (log(x*(1+4*x*(1+2*x))))/6 + log(pi)/2

nr = np.vectorize(r)

def ramanujan_logpmf_vectorproof(x, mu):
    """ Calculates log Poisson pmf using Ramanujan log-gamma approximation function
        r for 1 x and 1 mu.
        set up so it can be vectorized and take arrays
        of xs and mus

        :param x: random variable at which to calculate log probability
        :param mu: mean number of successes at random variable x
        :return: log probability that random variable x will happen
    """
    return x*math.log(mu) - nr(x) - mu

nramanujan_logpmf = np.vectorize(ramanujan_logpmf_vectorproof)

def ramanujan_pmf(x, mu):
    """Calculates Poisson pmf using Ramanujan log-gamma approximation function
        r for 1 x and 1 mu.
        set up so it can be vectorized and take arrays
        of xs and mus

        :param x: random variable at which to calculate probability
        :param mu: mean number of successes at random variable x
        :return: probability that random variable x will happen
    """
    y = nramanujan_logpmf(x,mu)
    return np.exp(y)

#mus is an array of mu values, one for each six_sensors

def probability_array_gen(pmfs, positions):
    """Instantiates an array of ones that is the correct shape
        for joint Poisson probability dist.

        :param pmfs: list of lists of pmfs that will help determine shape
        :param positions: array of radial positions, one for each sensor,
            the length will be last dimension of joint probability dist.
        :return: multidimensional array with an axis for each pmf in pmfs and
            one that is as long as the number of positions(?)
    """
    dimension_of_grid = ([len(p) for p in pmfs]) #first dimension, to start
    dimension_of_grid.append(len(positions))
    probabilities = zarr.ones((dimension_of_grid),chunks=(100,100))
    return probabilities

def poisson_integration(mus, known_vals, positions):
    """Integrates over a poisson distribution.
        Calculates mu values for broken sensors.
        Returns array of probabilities integrated over position

        :param mus: array of mu values for working sensors
        :param known_vals: array of intensities detected by working sensors,
            or -1s to indicate broken sensors
        :param positions: array of radial positions, one for each sensor
        :var
        :return: array of probabilities integrated over position
    """
# Saves indices of broken sensors, and saves probabilities of working sensors
    #at index of the intensity detected
    broken_sensor = np.empty(7)
    broken_count = 0
    for i, value in enumerate(known_vals):
        if value == -1:
            broken_sensor[broken_count] = i
            broken_count += 1

    #max_k for computing pmfs: starting upper limit for pmf gen
    #arbitrarily capping things at 410
    pmfs = []
    for i in range(broken_count):
        max_k = int(np.max(known_vals)/2)
        mu = (positions[int(broken_sensor[i])]-1)/.25
        # print("Mu", mu, "broken #", i)
        x = np.arange(0, max_k, 1)
        pmf = np.array(ramanujan_pmf(x,mu))
        old_max_k = max_k
        max_k += 10
        # print("x min", np.min(x), "x max", np.max(x))
        while np.sum(pmf) <= 0.9999 and max_k < 400:
            x = np.arange(old_max_k, max_k, 1, dtype=int)
            # print("x min", np.min(x), "x max", np.max(x))
            pmf = np.concatenate((pmf, ramanujan_pmf(x,mu))) #Look to see if this is a bottleneck later
            old_max_k = max_k
            max_k += 10
        pmfs.append(pmf)

#Here is where we are struggling:
#Trying to create joint dist.
    probabilities = probability_array_gen(pmfs, positions)
    temp = zarr.ones(([len(p) for p in pmfs]))
    shapes = np.ones([len(pmfs),len(pmfs)], dtype=int)
    np.fill_diagonal(shapes, [len(p) for p in pmfs])
    for i in range(broken_count):
        print(i)
        temp *= (np.array(pmfs[i])).reshape(shapes[i])
    print("This is what I thought the joint distribution should look like:")
    print(temp)
    dimension_of_grid = ([len(p) for p in pmfs]) #first dimension, to start
    dimension_of_grid.append(len(positions))
    probabilities *= np.expand_dims(temp,broken_count) #Doesn't work!
    np.exp(probabilities)
    print("Joint distribution in array with position dimension. Looks wrong.")
    print(probabilities)
    # probabilities = xr.DataArray(probabilities)
#To sum over sensor axes
    # probabilities = xr.DataArray(probabilities)
    # xr.IndexVariable.sum(probabilities, dim='dim_'+str(broken_count))
    for i in range(broken_count):
        probabilities = np.sum(probabilities,axis=0)
        probabilities = probabilities/np.sum(probabilities) #?
    return probabilities

positions = np.array([4,5])
mu_vals= np.array([6,0])
known_vals = np.array([13,-1])

print("?",poisson_integration(mu_vals,known_vals,positions))
