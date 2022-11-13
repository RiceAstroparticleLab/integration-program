import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit
from math import log, pi
import xarray as xr
import zarr

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

def probability_array_gen(pmfs, interaction_positions, broken_count, pmf_lengths):
    """Instantiates an array of ones that is the correct shape
        for joint Poisson probability dist.

        :param pmfs: list of lists of pmfs that will help determine shape
        :param positions: array of radial positions, one for each sensor,
            the length will be last dimension of joint probability dist.
        :return: multidimensional array with an axis for each pmf in pmfs and
            one that is as long as the number of positions(?)
    """

    dimension_of_grid = ([x for x in pmf_lengths]) #first dimension, to start
    print(dimension_of_grid)
    dimension_of_grid.append(interaction_positions)
    probabilities = zarr.ones((dimension_of_grid))
    print("Shape:", probabilities.shape)
    print("Chunks", probabilities.chunks)
    return probabilities

def calculate_univariate(k, mu, interaction_positions, radial_position):
    """
    Parameters
	----------
	k : integer
		max k value to calculate the Poisson for
	mu : float

	Returns
	-------
	pmf : array-like
	shape 2D (k, max k)

	return pmf
    """
    new_mus = np.array(interaction_positions, dtype=float)
    max_ks = np.zeros(interaction_positions, dtype=int)
    max_ks[0] = int(np.max(known_vals)/2)
    new_mus[0] = mu
    x = np.arange(0, max_k, 1)
    pmf = np.array(ramanujan_pmf(x,mu))
    for i in range(interaction_positions):
        max_ks[i] = max_k
        mu = (i + radial_positions[int(broken_sensor[i])]-1)/.25
        new_mus[i] = mu
        old_max_k = max_k
        max_k += 10
        while np.sum(pmf) <= 0.9999 and max_k < 400:
            max_ks[i] = max_k #why?
            x = np.arange(old_max_k, max_k, 1, dtype=int)
            pmf = np.concatenate((pmf, ramanujan_pmf(x,mu))) #Look to see if this is a bottleneck later
            old_max_k = max_k
            max_k += 10
        pmf_lengths[i] = len(pmf)
        pmfs.append(pmf)

    return pmfs, max_ks, new_mus

def create_joint_dist(pmfs, broken_count, interaction_positions):
    temp = zarr.ones(([len(p) for p in pmfs]), dtype=float)
    shapes = np.ones([len(pmfs),len(pmfs)], dtype=int)
    np.fill_diagonal(shapes, [len(p) for p in pmfs])
    for i in range(broken_count):
        # for j in range(interaction_positions):
        temp *= (np.array(pmfs[i])).reshape(shapes[i])
    print("This is what I thought the joint distribution should look like:")
    print(temp.shape)
    dimension_of_grid = ([len(p) for p in pmfs]) #first dimension, to start
    dimension_of_grid.append(len(positions))
    probabilities *= np.expand_dims(temp,broken_count) #Doesn't work!
    np.exp(probabilities)
    print("Joint distribution in array with position dimension. Looks wrong.")
    # probabilities = xr.DataArray(probabilities)
#To sum over sensor axes
    # probabilities = xr.DataArray(probabilities)
    # xr.IndexVariable.sum(probabilities, dim='dim_'+str(broken_count))

    return probabilities

def calculate_cross_term(mu1, mu2, cross_mu, max_k1, max_k2):
    temp = -1*np.exp(cross_mu)
    for i in range(max(max_k1,max_k2)):
        temp += ((**(ramanujan(k[i])))*(**ramanujan(k[i])))
    return temp *= **(ramanujan(k[i])) * (mus[broken_num1,broken_num2]/(mus[broken_num1]*mus[broken_num2]))


def calculate_multivariate(cross_mus, interaction_positions, max_ks):
    """
    Returns
	-------
	pmf : array-like
	shape 2D (k0, k1)

	exp(mu01) * sum ....
	return pmf

    # mu1 = -1*np.exp(mus[i])
    """
    for j in range(interaction_positions):
        for i in range(broken_count): #best way probably depends on order of mus
            for p in range(broken_count):
                probabilities[] *= ((-mu1[i,p]**max_ks[i,j])/**(ramanujan(max_ks[i,j]))

def poisson_integration(known_vals, interaction_positions, radial_positions, cross_mus):
    """Integrates over a poisson distribution.
        Calculates mu values for broken sensors.
        Returns array of probabilities integrated over position

        :param mus: array of mu values for working sensors
        :param known_vals: array of floats that holds intensities detected by working sensors,
            or -1s to indicate broken sensors
        :param positions: array floats that holds potential interaction positions
        :param positions: array floats that holds radial positions, one for each sensor
        :var
        :return: array of probabilities integrated over position
    """
    np.fill_diagonal(cross_mus,(interaction_positions + radial_positions[int(broken_sensor[x])]-1)/.25 for x, value in enumerate(cross_mus.diagonal()))
    new_mus = np.array((cross_mus.diagonal()*(interaction_positions + radial_positions[int(broken_sensor[i])]-1)/.25), dtype=float)
     = (interaction_positions + radial_positions[int(broken_sensor[i])]-1)/.25 #for all mus
    broken_sensor = np.empty(7)
    broken_count = 0
    for i, value in enumerate(known_vals):
        if value == -1:
            broken_sensor[broken_count] = i
            broken_count += 1

    # pmfs, max_ks, new_mus = calculate_univariate(mus, known_vals, interaction_positions, radial_positions)
    # probabilities = probability_array_gen(pmfs, interaction_positions, broken_count, pmf_lengths)
    max_ks = np.ones(broken_count)*400
    #look for viable base-cases:first need k[1], k[0] is calculated in the first loop
    univariate_poisson, max_ks[1] = calculate_univariate(cross_mus[1,1], interaction_positions, radial_position[1])
    probabilities = np.ones(tuple(max_ks),interaction_positions)
    for i in range(broken_count):
        univariate_poisson, max_ks[i], cross_mus[i,i] = calculate_univariate(cross_mus[i,i], interaction_positions, radial_position[i]) #change to calculate log-probabilities
        newaxes = [1] * broken_count
        newaxes[i] = [max_ks[i]]
        probabilities *= np.expand_dims(univariate_poisson, axis=tuple(newaxes))
        for j in np.arange(i+1, broken_count):
            cross_term = calculate_cross_term(cross_mus[i,i], cross_mus[j,j], cross_mus[i,j], max_ks[i], max_ks[j])
            newaxes = [1] * broken_count
            newaxes[i] = [max_ks[i]]
            newaxes[j] = [max_ks[j]]
            probabilities *= np.expand_dims(cross_term, axis=tuple(newaxes))



    for i in range(broken_count):
        probabilities = np.sum(probabilities,axis=0)
        probabilities = probabilities/np.sum(probabilities) #?
mus = np.array(([20,30,32],[31,12,8,][94,32,1]), dtype=float) #mu[0,0] = mu 0, mu[1,1] = mu 1... mu[0,1] = mu (0,1)...
interaction_positions = 5
radial_positions = np.array([4,5,7])
# base_mu_vals= np.array([6,4,7])
known_vals = np.array([13,-1,-1])
poisson_integration(known_vals, interaction_positions, radial_positions, mus)
