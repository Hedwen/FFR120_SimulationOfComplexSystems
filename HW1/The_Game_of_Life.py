#%% Import
import numpy as np
import matplotlib.pyplot as plt

#%% Functions

def GetNeighbors(status):

    # Initialize the neighbor count array
    n_nn = (
        np.roll(status, 1, axis=0) +  # Up.
        np.roll(status, -1, axis=0) +  # Down.
        np.roll(status, 1, axis=1) +  # Left.
        np.roll(status, -1, axis=1) +  # Right.
        np.roll(np.roll(status, 1, axis=0), 1, axis=1) +  # Up-Left.
        np.roll(np.roll(status, 1, axis=0), -1, axis=1) +  # Up-Right
        np.roll(np.roll(status, -1, axis=0), 1, axis=1) +  # Down-Left
        np.roll(np.roll(status, -1, axis=0), -1, axis=1)  # Down-Right
    )

    return n_nn

def ApplyRule(rule_2d, status):
    Ni, Nj = status.shape
    next_status = np.zeros([Ni, Nj])

    n_neighbors = GetNeighbors(status)
    for i in range(Ni):
        for j in range(Nj):
            next_status[i,j] = rule_2d[int(status[i,j]), int(n_neighbors[i,j])]
    
    return next_status

#%% Simulation
N = 100     # Bigger if possible

n_runs = 5
n_time_steps = 1000

steady_time_steps = 200

alive_array = np.zeros([n_runs, n_time_steps])
change_array = np.zeros([n_runs, n_time_steps-steady_time_steps])

for runs in range(n_runs):
    print(f'Starting run {runs+1}')

    gol = np.random.randint(2, size=[N, N])

    rule_2d = np.zeros([2, 9])
    rule_2d[0, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0] # Dead cell
    rule_2d[1, :] = [0, 0, 1, 1, 0, 0, 0, 0, 0] # Avlive cell

    Ni, Nj = gol.shape

    for step in range(n_time_steps):
        new_gol = ApplyRule(rule_2d, gol)

        if step >= steady_time_steps:
            change_array[runs,step-steady_time_steps] = np.sum(gol != new_gol)

        alive_array[runs,step] = np.sum(new_gol)

        gol = new_gol

#%% Plot
t = np.arange(n_time_steps)

for i in range(len(alive_array)):
    plt.plot(t, alive_array[i,:], '.-')
    plt.hlines(np.mean(alive_array[i,:]),t[0],t[-1],colors='k', linestyles='--')
    plt.xlabel(r'$t$')
    plt.ylabel(f'Number of live cells')
    plt.title(f'Number of live cells as a function of time for run {i+1}')
    plt.grid()
    plt.show()

    plt.plot(t[steady_time_steps:], change_array[i,:], '.-')
    plt.xlabel(r'$t$')
    plt.ylabel(f'Number of changes')
    plt.title(f'Number of changes in cells as a function of time for run {i+1}')
    plt.grid()
    plt.show()

avg_density = np.mean(alive_array)/gol.size

print(f'Average density of alive cells: {avg_density}')

for i in range(len(alive_array)):
    avg = np.mean(alive_array[i,:])
    print(f'Average from run {i+1}: {avg}')


# %%
