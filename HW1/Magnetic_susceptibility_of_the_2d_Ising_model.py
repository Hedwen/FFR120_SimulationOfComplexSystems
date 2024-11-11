#%% Import
import numpy as np
import matplotlib.pyplot as plt

#%% Functions

def IsingModelStep(i0, j0, sl):
    M = sl[i0-1,j0] + sl[i0+1,j0] + sl[i0,j0-1] + sl[i0,j0+1]

    E_pos = -(H + J*M)
    E_neg = H + J*M



#%% Parameters

N = 100     # Bigger if possible
J = 1
kb = 1

#%% Task 1
H_list = [-5, -2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 5]
T = 5

spin_lattice = 2 * np.random.randint(2, size=(N,N)) - 1





#%% Task 2
T_list = [0.1, 0.2, 0.5, 1, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 5]
H = 0

max_iterations = 5000