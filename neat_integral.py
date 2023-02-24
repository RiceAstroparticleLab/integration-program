import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit
from math import log, pi
# import xarray as xr
import zarr
import scipy
from scipy import special

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

def probability_array_gen(interaction_positions, broken_count, max_ks):
    """Instantiates an array of ones that is the correct shape
        for joint Poisson probability dist.

        :param pmfs: list of lists of pmfs that will help determine shape
        :param positions: array of radial positions, one for each sensor,
            the length will be last dimension of joint probability dist.
        :return: multidimensional array with an axis for each pmf in pmfs and
            one that is as long as the number of positions(?)
    """

    dimension_of_grid = ([max_ks[x] for x in range(broken_count)]) #first dimension, to start
    dimension_of_grid.append(interaction_positions)
    # probabilities = zarr.array((np.nan(dimension_of_grid))) #change to fill with negative infinities
    probabilities = zarr.ones((dimension_of_grid))
    # probabilities *= numpy.nan
    # probabilities[:] *= -math.inf
    return probabilities

def calculate_univariate(mu, interaction_positions, radial_position):
    """
    Parameters
	----------
	mu : float

	Returns
	-------
	pmf : array-like
	shape 2D (k, max k)

	return pmf
    """
    pmfs = []
    max_k = 10 #ask Tina what to set this as!
    x = np.arange(0, max_k, 1)
    pmf = np.array(nramanujan_logpmf(x,mu))
    for i in range(interaction_positions):
        max_k = 10
        old_max_k = 0
        x = np.arange(0, max_k, 1)
        pmf = np.array(nramanujan_logpmf(x,mu))
        while np.sum(pmf) <= 0.9999 and max_k < 100:
            x = np.arange(old_max_k, max_k, 1, dtype=int)
            pmf = np.concatenate((pmf, nramanujan_logpmf(x,mu))) #Look to see if this is a bottleneck later
            old_max_k = max_k
            max_k += 10
        pmfs.append(pmf)
        #must make max_k the longest of the pmfs per interaction position
    return pmfs, max_k

def create_joint_dist(pmfs, broken_count, interaction_positions, probabilities, max_ks):
    """
    pmfs :

    """
    for i in range(broken_count):
        for j in range(interaction_positions):
            newaxes = [1] * broken_count
            newaxes[i] = int(len(pmfs[i][j]))
            # if values are finite, do this, else do that
            # if probabilities[0,0,j] == nan:
            #     probabilities[:,:,j] = np.array(pmfs[i][j]).reshape(newaxes)
            # else:
            probabilities[:,:,j] += np.array(pmfs[i][j]).reshape(newaxes)
    return probabilities

            # probabilities[:,:,j] *= (np.array(pmfs[(i * interaction_positions) +j])).reshape(shapes[i]) #doing the reshaping to pmfs in temp


def calculate_cross_term(mu1, mu2, cross_mu, max_k1, max_k2):
    """
    mu1 : float, mu value of same sensor max_k1 corresponds to
    mu2 : float, mu value of same sensor max_k2 corresponds to
    cross_mu : float
    max_k1 : int
    max_k2 : int

    Returns
    -------
    float
    """
    temp = 0
    temp1 = 0
    first = 0
    second = 0
    third = 0
    for p in range(min(max_k1,max_k2)):
# Calculate cross term
        # temp += np.exp(nr(max_k1) + nr(max_k2) - nr(p) - nr(max_k1 - p) - nr(max_k2 - p)) * (np.exp(nr(p)) * (cross_mu/(mu1*mu2)))
        # print("Normal cross:", temp)
# Modified cross term
        temp1 += np.exp(1j*math.pi*min(max_k1,max_k2)) * ((cross_mu/(mu1*mu2))**min(max_k1,max_k2))*scipy.special.hyperu(-min(max_k1,max_k2),-min(max_k1,max_k2)+max(max_k1,max_k2)+1,-((mu1*mu2)/cross_mu))
        first += np.exp(1j*math.pi*min(max_k1,max_k2))
        second += ((cross_mu/(mu1*mu2))**min(max_k1,max_k2))
        third += scipy.special.hyperu(-min(max_k1,max_k2),-min(max_k1,max_k2)+max(max_k1,max_k2)+1,-((mu1*mu2)/cross_mu))

        print("First part", first)
        print("Second part", second)
        print("Third part", third)
        print("New cross:", temp1)
    return float(math.log(temp1)) + (cross_mu)


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
                probabilities[i,p,j] *= ((-mu1[i,p]**max_ks[i,j])/np.exp(nr(max_ks[i,j])))

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

    broken_sensor = []
    broken_count = 0
    for i, value in enumerate(known_vals):
        if value == -1:
            broken_sensor.append(i)
            broken_count += 1

    #line below is meant to calculate new mus and correctly replace them in array cross_mus
    # np.fill_diagonal(cross_mus,(interaction_positions + radial_positions[int(broken_sensor[x])]-1)/.25 for x, value in enumerate(cross_mus.diagonal()))
    for i in range(broken_count):
        cross_mus[broken_sensor[i],broken_sensor[i]] = (interaction_positions + radial_positions[int(i)]-1)/.25

    # pmfs, max_ks, cross_mus = calculate_univariate(np.diagonal(cross_mus), known_vals, interaction_positions, radial_positions)

    max_ks = np.empty(broken_count)
    univariate_poissons = []

    for i in range(broken_count):
        univariate_poisson, max_ks[i] = calculate_univariate(cross_mus[broken_sensor[i],broken_sensor[i]], interaction_positions, radial_positions[broken_sensor[i]]) #change to calculate log-probabilities
        univariate_poissons.append(univariate_poisson)
        newaxes = [1] * broken_count
        newaxes[i] = [max_ks[i]]
    univariate_poissons = np.array(univariate_poissons)
    probabilities = probability_array_gen(interaction_positions, broken_count, max_ks) #save univariate_poissons to list to pass in for pmfs parameter
    probabilities = np.exp(create_joint_dist(univariate_poissons, broken_count, interaction_positions, probabilities, max_ks))
    for i in range(broken_count):
        for j in np.arange(i+1, broken_count):
            cross_term = calculate_cross_term(cross_mus[broken_sensor[i],broken_sensor[i]], cross_mus[broken_sensor[j],broken_sensor[j]], cross_mus[broken_sensor[i],broken_sensor[j]], int(max_ks[i]), int(max_ks[j]))
            newaxes = [1] * broken_count
            newaxes[i] = int(max_ks[i])
            newaxes[j] = int(max_ks[j])
            print("cross", cross_term)
            # probabilities[i,j] *= np.array(cross_term).reshape(newaxes)
            probabilities[i,j] *= np.array(cross_term)

 # np.array(pmfs[i][j]).reshape(newaxes)
#calculate integral
    for i in range(broken_count):
        probabilities = np.sum(probabilities,axis=-1)
        probabilities = probabilities/np.sum(probabilities) #renormalizing
    return probabilities

mus = np.array(([20,30,32],[31,12,8],[94,32,1]), dtype=float) #mu[0,0] = mu 0, mu[1,1] = mu 1... mu[0,1] = mu (0,1)...
interaction_positions = 5
radial_positions = np.array([4,5,7])
# base_mu_vals= np.array([6,4,7])
known_vals = np.array([13,-1,-1])
print(poisson_integration(known_vals, interaction_positions, radial_positions, mus))
