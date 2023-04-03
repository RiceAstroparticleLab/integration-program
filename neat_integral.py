import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit
from math import log, pi
# import xarray as xr
import zarr
import scipy
from scipy import special
import sys
import warnings
warnings.filterwarnings("error")


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

def probability_array_gen(interaction_positions, broken_count, max_ks, broken_sensor):
    """Instantiates an array of ones that is the correct shape
        for joint Poisson probability dist.

        :param pmfs: list of lists of pmfs that will help determine shape
        :param positions: array of radial positions, one for each sensor,
            the length will be last dimension of joint probability dist.
        :return: multidimensional array with an axis for each pmf in pmfs and
            one that is as long as the number of positions(?)
    """
    print(broken_count)
    dimension_of_grid = ([max_ks[broken_sensor[x]] for x in range(broken_count)]) #first dimension, to start
    dimension_of_grid.append(len(interaction_positions))
    # probabilities = zarr.array((np.nan(dimension_of_grid))) #change to fill with negative infinities
    print("dimension_of_grid", dimension_of_grid)
    probabilities = zarr.ones((dimension_of_grid))
    # probabilities *= numpy.nan
    # probabilities[:] *= -math.inf
    return probabilities

def calculate_univariate(mus, interaction_positions, radial_position):
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
    pmf = np.array(nramanujan_logpmf(x,mus[0]))
    for i in range(len(interaction_positions)):
        max_k = 10
        old_max_k = 0
        x = np.arange(0, max_k, 1)
        pmf = np.array(nramanujan_logpmf(x,mus[i]))
        while np.sum(pmf) <= 0.9999 and max_k < 100:
            x = np.arange(old_max_k, max_k, 1, dtype=int)
            pmf = np.concatenate((pmf, nramanujan_logpmf(x,mus[i]))) #Look to see if this is a bottleneck later
            old_max_k = max_k
            max_k += 10
        pmfs.append(pmf)
        #must make max_k the longest of the pmfs per interaction position
    return pmfs, max_k

def create_joint_dist(pmfs, broken_count, interaction_positions, probabilities, max_ks):
    print("broken count in joint dist.", broken_count)
    """
    pmfs :

    """
    print("probabilities shape", probabilities.shape)
    for i in range(broken_count):
        for j, value in enumerate(interaction_positions):
            newaxes = [1] * broken_count
            newaxes[i] = int(len(pmfs[i][j]))
            print("new axes in loop", newaxes, "pmfs ", pmfs[i][j].shape)
            probabilities[...,j] += np.array(pmfs[i][j]).reshape(newaxes)
    # print("final new axes", nexaxes)
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

    print(sys.float_info.min)
    print("max_k1", )

    cross_term = np.exp(cross_mu) * (-1.*(mu1*mu2)/cross_mu)**(-1.*min(max_k1,max_k2))
    cross_term *= scipy.special.hyp1f1(-min(max_k1,max_k2),-min(max_k1,max_k2)+max(max_k1,max_k2)+1,-((mu1*mu2)/cross_mu))
    print("cross_term", cross_term)
    try:
        np.log(cross_term)
    except RuntimeWarning:
        cross_term[cross_term < sys.float_info.min] = sys.float_info.min
        print("cross_terms in except", cross_term)
    return (np.log(cross_term)) + (cross_mu) #getting log prob. of the summation term in cross - mu equation, then adding to log(e^mu) to get cross-mu


# def calculate_multivariate(cross_mus, interaction_positions, max_ks):
#     """
#     Returns
# 	-------
# 	pmf : array-like
# 	shape 2D (k0, k1)
#
# 	exp(mu01) * sum ....
# 	return pmf
#
#     # mu1 = -1*np.exp(mus[i])
#     """
#     for j, value in enumerate(interaction_positions):
#         for i in range(broken_count): #best way probably depends on order of mus
#             for p in range(broken_count):
#                 probabilities[i,p,j] *= ((-mu1[i,p]**max_ks[i,j])/np.exp(nr(max_ks[i,j])))

def poisson_integration(values, interaction_positions, radial_positions, cross_mus):
    """Integrates over a poisson distribution.
        Calculates mu values for broken sensors.
        Returns array of probabilities integrated over position

        :param mus: array of mu values for working sensors
        :param values: array of floats that holds intensities detected by working sensors,
            or -1s to indicate broken sensors
        :param interaction_positions: array floats that holds potential radial interaction positions
        :param radial_positions: array floats that holds radial positions, one for each sensor
        :var
        :return: array of probabilities integrated over position
    """
    known_vals = []
    broken_sensor = []
    broken_count = 0
    for i, value in enumerate(values):
        if value == -1.:
            broken_sensor.append(i)
            broken_count += 1
        else:
            known_vals.append(i)
    print("broken_count", broken_count)
    #line below is meant to calculate new mus and correctly replace them in array cross_mus

    # pmfs, max_ks, cross_mus = calculate_univariate(np.diagonal(cross_mus), known_vals, interaction_positions, radial_positions)

    max_ks = np.empty(len(values))
    univariate_poissons = []

    for i in range(broken_count):
        univariate_poisson, max_ks[broken_sensor[i]] = calculate_univariate(cross_mus[broken_sensor[i],broken_sensor[i],:], interaction_positions, radial_positions[broken_sensor[i]]) #change to calculate log-probabilities
        univariate_poissons.append(univariate_poisson)
        # newaxes = [1] * broken_count
        # newaxes[i] = [max_ks[i]]

    print(univariate_poisson)

    for i, value in enumerate(known_vals):
        max_ks[known_vals[i]] = values[known_vals[i]]
        # newaxes = [1] * broken_count
        # newaxes[i] = [max_ks[i]]

# TO-DO: for known_vals, essentially skip the above loop, use the known k instead of
# finding the max k, still calculate the bivariate etc...
    print("max_ks", max_ks)
    univariate_poissons = np.array(univariate_poissons)
    probabilities = probability_array_gen(interaction_positions, broken_count, max_ks, broken_sensor) #save univariate_poissons to list to pass in for pmfs parameter
    probabilities = np.exp(create_joint_dist(univariate_poissons, broken_count, interaction_positions, probabilities, max_ks))

    if broken_count > 1: #need multiple versions of below code for multiple
        for i in range(broken_count):
            for j in np.arange(i+1, broken_count):
                print("1", cross_mus[broken_sensor[i],broken_sensor[i]], "2", cross_mus[broken_sensor[j],broken_sensor[j]], "3", cross_mus[broken_sensor[i],broken_sensor[j]], "4",  int(max_ks[i]), "5", int(max_ks[j]))
                cross_term = calculate_cross_term(cross_mus[broken_sensor[i],broken_sensor[i]], cross_mus[broken_sensor[j],broken_sensor[j]], cross_mus[broken_sensor[i],broken_sensor[j]], int(max_ks[i]), int(max_ks[j]))
                probabilities[i,j] *= np.array(cross_term) #probabilities indexing may need work

    for i in range(broken_count):
        probabilities = np.sum(probabilities,axis=-1)
        probabilities = probabilities/np.sum(probabilities) #renormalizing
    return probabilities

# mus = np.array(([20,30,32],[31,12,8],[94,32,1]), dtype=float) #mu[0,0] = mu 0, mu[1,1] = mu 1... mu[0,1] = mu (0,1)...
# interaction positions are possible radial positions where the interaction could have occured
# calculating probability that interaction could have occurred at each position in array. Shape of final output array is affected by this var.


radial_positions = [ 0., 8.01,  8.01,  8.01, 13.88, 13.88, 16.02]
values = [185., -1., 358., -1.,  44., -1.,  75.]
interaction_positions = [0.7,  1.9,  2.9]
cross_mus = np.ndarray((len(values), len(values), len(interaction_positions)))
for i, value in enumerate(values):
    for p, value in enumerate(values):
        for j, value in enumerate(interaction_positions):
            cross_mus[i,p,j] = (interaction_positions[j] + radial_positions[int(i)]-1)/.25

f = open("p_integral_test.txt", "w")
f.write("Test with radial positions: ")
f.write(np.array2string(np.array(radial_positions)))
f.write("\nvalues: ")
f.write(np.array2string(np.array(values)))
f.write("\ninteraction positions: ")
f.write(np.array2string(np.array(interaction_positions)))
poisson = poisson_integration(values, interaction_positions, radial_positions, cross_mus)
f.write("\noutput: ")
f.write(np.array2string(poisson))
print(poisson)
f.write("\nsum of distribution: ")
f.write(np.array2string(np.sum(poisson)))
