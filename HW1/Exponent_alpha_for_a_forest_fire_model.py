#%% Import
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import *

#%% Functions
def GrowTrees(forest, p):
    Ni, Nj = forest.shape

    new_trees = np.random.rand(Ni, Nj)
    new_trees_idx = np.where(new_trees <= p)

    forest[new_trees_idx] = 1
    return forest

def PropagateFire(forest, i0, j0):
    Ni, Nj = forest.shape

    fs = 0

    if forest[i0, j0] == 1:
        active_i = [i0]
        active_j = [j0]

        forest[i0, j0] = -1
        fs += 1

        while len(active_i) > 0:
            next_i = []
            next_j = []

            for n in np.arange(len(active_i)):
                forest, fs, next_i_n, next_j_n = SpreadingFire(forest, fs, active_i[n], active_j[n], Ni, Nj)

                next_i += next_i_n
                next_j += next_j_n
            
            active_i = next_i
            active_j = next_j
    
    return forest, fs

def SpreadingFire(forest, fs, active_i, active_j, Ni, Nj):
    next_i = []
    next_j = []

    # Up
    i = (active_i + 1) % Ni
    j = active_j
    # Check Status
    if forest[i,j] == 1:
        next_i.append(i)
        next_j.append(j)
        forest[i,j] = -1
        fs += 1
    
    # Down
    i = (active_i - 1) % Ni
    j = active_j
    # Check Status
    if forest[i,j] == 1:
        next_i.append(i)
        next_j.append(j)
        forest[i,j] = -1
        fs += 1
    
    # Left
    i = active_i
    j = (active_j - 1) % Nj
    # Check Status
    if forest[i,j] == 1:
        next_i.append(i)
        next_j.append(j)
        forest[i,j] = -1
        fs += 1

    # Right
    i = active_i
    j = (active_j + 1) % Nj
    # Check Status
    if forest[i,j] == 1:
        next_i.append(i)
        next_j.append(j)
        forest[i,j] = -1
        fs += 1

    return forest, fs, next_i, next_j

def RunForestFireSimulation(forest, p, f, target_num_fires):

    fire_size = []
    num_fires = 0

    while num_fires < target_num_fires:
        forest = GrowTrees(forest, p)

        if np.random.rand() <= f:
            i0 = np.random.randint(0,len(forest))
            j0 = np.random.randint(0,len(forest[0]))

            forest, fs = PropagateFire(forest, i0, j0)

            if fs > 0:
                fire_size.append(fs)
                num_fires += 1

        forest[np.where(forest == -1)] = 0

    return fire_size

def complementary_CDF(f, f_max):
    s_rel = np.sort(np.array(f)) / f_max
    c_CDF = np.array(np.arange(len(f), 0, -1)) / len(f)
    return c_CDF, s_rel

def CalculateAlpha(fire_size, forest_size):
    c_CDF, s_rel = complementary_CDF(fire_size, forest_size)

    min_rel_size = 0.001
    max_rel_size = 0.1

    idx_min = np.searchsorted(s_rel, min_rel_size)
    idx_max = np.searchsorted(s_rel, max_rel_size)

    p = np.polyfit(np.log(s_rel[idx_min:idx_max]),
                   np.log(c_CDF[idx_min:idx_max]), 1)
    
    alpha = 1 - p[0]
    return alpha


#%% Parameters
N_list = [16, 32, 64, 128, 256, 512, 1024]
p = 0.01
f = 0.2

num_measurements = 10

target_num_fires = 300

alpha_array = np.zeros([len(N_list),num_measurements])

for i,N in enumerate(N_list):
    print(f'Forest size: {N}')
    for j in range(num_measurements):
        print(f'Measurement: {j}')
        forest = np.zeros([N,N])

        fire_size = RunForestFireSimulation(forest, p, f, target_num_fires)

        alpha_array[i,j] = CalculateAlpha(fire_size, forest.size)

# %% Plot

alpha_average = np.mean(alpha_array, axis=1)
alpha_std = np.std(alpha_array, axis=1)

N_inverse = [1 / N for N in N_list]

# Alpha at 1/N = 0
p = np.polyfit(N_inverse[1:], alpha_average[1:], 1)
alpha_inf = p[1]

plt.errorbar(N_inverse, alpha_average, yerr=alpha_std, fmt='o')
plt.plot(N_inverse, p[0]*np.array(N_inverse) + p[1], '--k')
plt.ylim(1.09,1.35)
plt.xlabel(r'$1/N$')
plt.ylabel(r'$\alpha$')
plt.title(r'Exponent $\alpha_N$ as a function of $1/N$')
plt.grid()
plt.show()

print(f'Calculated value of alpha: {alpha_inf}')

# %%
