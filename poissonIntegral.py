import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit
from math import log, pi
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
    probabilities = np.ones((dimension_of_grid))
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

    probabilities = probability_array_gen(pmfs, positions)

#Here is where we are struggling:
#Trying to create joint dist.
#Want something that does this but quicker:
        # for k_1 in pmfs[0]:
        #     for k_2 in pmfs[1]:
        #         for k_3 in pmfs[2]:
        #             probabilities[k_1,k_2,k_3] = k_1*k_2*k_3

    pmfs = [[.25,.20,.20,.15],[.24,.21,.19,.16],[.23,.42,.35]] #temporary pmfs list

#Attempts
    # np.broadcast_to()
    # temp = np.broadcast(pmfs[i], probabilities[:,:])
    # probabilities = np.empty(probabilities.shape)
    # print("Here",([p] for p in pmfs))
    # probabilities = einsum()
    # probabilities.flat = [u*v for (u,v) in temp] #can't multiply more than two? How to automatically fill in generation func
    # print("Pmfs shape", len(pmfs[0]))
    # print("Probability shape", probabilities.shape)
    # probabilities = np.tile()

    temp = np.array(np.meshgrid([p for p in pmfs[0]],[p for p in pmfs[1]],[p for p in pmfs[2]])).T.reshape(-1,4,4,3)
    print((p for p in temp))
    print(temp)
    # temp.reshape(4,4,3,4)
    # print("^ before multiplying. v after multiplying")
    print("shape", temp.shape)
    print(np.tensordot(temp[0],temp[1], axes=([1,0],[0,1])))
    # print(np.prod(temp, axis = 1))

    print(probabilities)

#To sum over sensor axes
    for i in range(broken_count):
        probabilities = np.sum(probabilities,axis=0)
        probabilities = probabilities/np.sum(probabilities) #?
    return np.sum(probabilities)

    #Mix of tile and repeat, then reshape? maybe just to compare for right answer
    #Einstein's product

positions = np.array([4,8,6])
mu_vals= np.array([6,0,0])
known_vals = np.array([13,-1,-1])
# positions = [4,6]
# mu_vals=[6,0]
# known_vals=[13,-1]



poisson_integration(mu_vals,known_vals,positions)

# print(np.sum(grid))
# np.random.seed(19680801)
# # Z = np.random.rand(4, 13)
# # Z = np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0]])
# Z = grid
# y = np.arange(0,max_k_vals[0]+1,1,dtype=int)  # len = 11
# x = np.arange(0,max_k_vals[1]+1,1,dtype=int)  # len = 7
# fig, ax = plt.subplots()
# ah = ax.pcolormesh(x, y, Z)
# # cf = ax1.contourf(x[:-1, :-1] + dx/2.,
# #                   y[:-1, :-1] + dy/2., z, levels=levels,
# #                   cmap=cmap)
# fig.colorbar(ah, label='Probability')
# plt.title("Joint Probability Distribution of Two Broken Sensors")
# plt.xlabel("Poisson Pmf of a Broken Sensor with Mu = 12")
# plt.ylabel("Poisson Pmf of a Broken Sensor with Mu = 7")
# plt.show()
