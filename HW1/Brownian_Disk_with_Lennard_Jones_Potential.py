#%% Import
import numpy as np
import matplotlib.pyplot as plt
import math

#%% Functions

def InitializeParticles(X_disk, Y_disk, sigma, N_part, v_part, x_min, x_max, y_min, y_max):
    # Initialize particles positions
    x0, y0 = np.meshgrid(
        np.linspace(x_min, x_max, N_part),
        np.linspace(y_min, y_max, N_part))
    x0 = x0.flatten()
    y0 = y0.flatten()

    # Remove particles inside/too close to disk
    x0_temp = []
    y0_temp = []
    for i in range(len(x0)):
        r = np.sqrt((X_disk - x0[i])**2 + (Y_disk - y0[i])**2)
        if r > R_disk + 3*sigma:
            x0_temp.append(x0[i])
            y0_temp.append(y0[i])
    x0 = np.array(x0_temp)
    y0 = np.array(y0_temp)

    # Initialize particles velocities
    velocities = np.zeros((2,len(x0)))
    for i in range(len(velocities[0])):
        angle = np.random.uniform(0, 2*math.pi)
        velocities[:,i] = [v_part*np.cos(angle), v_part*np.sin(angle)]

    return x0, y0, velocities

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

#%% Simulation

x0, y0, velocities = InitializeParticles(X_disk, Y_disk, sigma, N_part, v_part,
                                         x_min, x_max, y_min, y_max)

N_t = int(T_tot/delta_t)
diskPosition = np.zeros((2,N_t))

for n_t in range(N_t):
    print(f'Starting measurement {n_t}')
    diskPosition[:,n_t] = [X_disk, Y_disk]

    dx, dy = X_disk - x0, Y_disk - y0

    r = np.sqrt((dx)**2 + (dy)**2) - R_disk

    # Calculate forces
    Fx_disk, Fy_disk = 0, 0
    for i in range(len(x0)):
        if r[i] > 0:
            F = 24 * epsilon/r[i] * (2*(sigma/r[i])**12 - (sigma/r[i])**6)

            Fx, Fy = F * dx[i] / (r[i] + R_disk), F * dy[i] / (r[i] + R_disk)

            Fx_disk += Fx
            Fy_disk += Fy

            # Update particle velocities
            velocities[0,i] += Fx / m_part * delta_t
            velocities[1,i] += Fy / m_part * delta_t

    # Update disk velocity
    v_disk[0] += np.sum(Fx_disk) / m_disk * delta_t
    v_disk[1] += np.sum(Fy_disk) / m_disk * delta_t

    # Update particle positions
    x0 += velocities[0,:] * delta_t
    y0 += velocities[1,:] * delta_t

    # Check box constraints
    for i in range(len(x0)):
        if x0[i] < x_min or x0[i] > x_max:
            velocities[0,i] = -velocities[0,i]
        if y0[i] < y_min or y0[i] > y_max:
            velocities[1,i] = -velocities[1,i]

    # Update disk position
    X_disk += v_disk[0] * delta_t
    Y_disk += v_disk[1] * delta_t

    # Check box constraints
    if X_disk < x_min+R_disk or X_disk > x_max-R_disk:
        v_disk[0] = -v_disk[0]
    if Y_disk < y_min+R_disk or Y_disk > y_max-R_disk:
        v_disk[1] = -v_disk[1]

# Calculate MSD
MSD = np.zeros((N_t))

for n_t in range(1,N_t):
    displacement = np.sum((diskPosition[:,n_t:N_t] - diskPosition[:,:N_t-n_t])**2)
    MSD[n_t-1] = np.mean(displacement)

#%% Plot
# Disk trajectory
plt.plot(diskPosition[0,:], diskPosition[1,:], '.')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.title("Trajectory of the Brownian Disk in the Cartesian plane")
plt.grid()
plt.show()

# MSD
timeSteps = np.arange(N_t) * delta_t
plt.plot(timeSteps, MSD)
plt.xlabel("Time")
plt.ylabel("Mean Square Displacement")
plt.title("Mean Square Displacement (MSD) of Disk")
plt.grid()
plt.show()

# %%
