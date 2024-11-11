#%% Import
import numpy as np
import matplotlib.pyplot as plt
import math

#%% Parameters

# Particle Parameters
m_part = 1
v_part = 10
N_part = 25

# Disk Parameters
m_disk = 10
R_disk = 10
X_disk = 0
Y_disk = 0
v_disk = [0,0]

# Box Parameter
L = 260
x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

# Lennard-Jones Parameters
sigma = 1
epsilon = 1

# Other Parameters
T_tot = 400
delta_t = 0.005

# Initialize particles positions
x0, y0 = np.meshgrid(
    np.linspace(x_min, x_max, N_part),
    np.linspace(y_min, y_max, N_part))
x0 = x0.flatten()
y0 = y0.flatten()

# Initialize particles velocities
velocities = np.zeros((2,N_part**2))
for i in range(len(velocities[0])):
    angle = np.random.uniform(0, 2*math.pi)
    velocities[:,i] = [v_part*np.cos(angle), v_part*np.sin(angle)]

# Remove particles inside/too close to disk

#%% Simulation

Y_diskDisplacement = 0
N_t = T_tot/delta_t
MSD = np.NaN((1,N_t))

T = 0
n_t = 0

while T < T_tot:
    T += delta_t
    n_t += 1

    r = np.sqrt((X_disk - x0)**2 + (Y_disk - y0)**2) - R_disk
    F = 24 * epsilon/r * (2*(sigma/r)**12 - (sigma/r)**6)

    dv_part = F/m_part





    X_diskNew = 0
    Y_diskNew = 0

    Y_diskDisplacement += (X_diskNew - X_disk)**2 + (Y_diskNew - Y_disk)**2
    MSD[n_t-1] = 1/(N_t-n_t)
    






# %%
