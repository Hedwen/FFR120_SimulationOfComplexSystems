#%% Import
import numpy as np
import matplotlib.pyplot as plt
import math

#%% Functions

def InitializeParticles(X_disk, Y_disk, sigma, N_part, v_part, x_min, x_max, y_min, y_max):
    # Initialize particles positions
    x0, y0 = np.meshgrid(
        np.linspace(x_min, x_max, int(np.sqrt(N_part))),
        np.linspace(y_min, y_max, int(np.sqrt(N_part))))
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

def CheckBoxConstraint(x, y, vx, vy, x_min, x_max, y_min, y_max):
    if x < x_min:
        vx = -vx
        x = 2*x_min - x
    elif x > x_max:
        vx = -vx
        x = 2*x_max - x

    if y < y_min:
        vy = -vy
        y = 2*y_min - y
    elif y > y_max:
        vy = -vy
        y = 2*y_max - y
    
    return x, y, vx, vy

#%% Parameters

# Particle Parameters
m_part = 1
v_part = 10
N_part = 625

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
    if n_t%500 == 0:
        print(f'Starting measurement {n_t}')

    diskPosition[:,n_t] = [X_disk, Y_disk]

    dx, dy = X_disk - x0, Y_disk - y0

    r = np.sqrt(dx**2 + dy**2) - R_disk

    # Calculate forces
    Fx_disk, Fy_disk = 0, 0
    for i in range(len(x0)):
        if r[i] > 0:
            F = 24 * epsilon/r[i] * (2*(sigma/r[i])**12 - (sigma/r[i])**6)

            Fx, Fy = F * dx[i] / r[i], F * dy[i] / r[i]

            Fx_disk += Fx
            Fy_disk += Fy

            # Update particle velocities
            velocities[0,i] += -Fx / m_part * delta_t
            velocities[1,i] += -Fy / m_part * delta_t

    # Update disk velocity
    v_disk[0] += Fx_disk / m_disk * delta_t
    v_disk[1] += Fy_disk / m_disk * delta_t

    # Update particle positions
    x0 += velocities[0,:] * delta_t
    y0 += velocities[1,:] * delta_t

    # Check box constraints
    for i in range(len(x0)):
        x0[i], y0[i], velocities[0,i], velocities[1,i] = CheckBoxConstraint(x0[i], y0[i], velocities[0,i], velocities[1,i], 
                                                                            x_min, x_max, y_min, y_max)

    # Update disk position
    X_disk += v_disk[0] * delta_t
    Y_disk += v_disk[1] * delta_t

    # Check box constraints
    X_disk, Y_disk, v_disk[0], v_disk[1] = CheckBoxConstraint(X_disk, Y_disk, v_disk[0], v_disk[1],
                                                              x_min+R_disk, x_max-R_disk, y_min+R_disk, y_max-R_disk)

#%% Calculate MSD
MSD = np.zeros((N_t))

for n_t in range(1,N_t):
    displacement = np.sum((diskPosition[:,n_t:N_t] - diskPosition[:,:N_t-n_t])**2)
    MSD[n_t-1] = np.mean(displacement)

#%% Plot
# Disk trajectory
plt.plot(diskPosition[0,:], diskPosition[1,:])
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

# %% Calculate estimate of D

MSD(tau) = 4*D*tau
