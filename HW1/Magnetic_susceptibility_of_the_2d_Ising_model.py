#%% Import
import numpy as np
import matplotlib.pyplot as plt

#%% Functions

def neighboring_spins(i_list, j_list, sl):

    Ni, Nj = sl.shape  # Shape of the spin lattice.
    
    # Position neighbors right.
    i_r = i_list  
    j_r = list(map(lambda x:(x + 1) % Nj, j_list))   

    # Position neighbors left.
    i_l = i_list  
    j_l = list(map(lambda x:(x - 1) % Nj, j_list))   

    # Position neighbors up.
    i_u = list(map(lambda x:(x - 1) % Ni, i_list))  
    j_u = j_list  

    # Position neighbors down.
    i_d = list(map(lambda x:(x + 1) % Ni, i_list)) 
    j_d = j_list   

    # Spin values.
    sl_u = sl[i_u, j_u]
    sl_d = sl[i_d, j_d]
    sl_l = sl[i_l, j_l]
    sl_r = sl[i_r, j_r]

    return sl_u, sl_d, sl_l, sl_r

def energies_spins(i_list, j_list, sl, H, J):

    sl_u, sl_d, sl_l, sl_r = neighboring_spins(i_list, j_list, sl)
    
    sl_s = sl_u + sl_d + sl_l + sl_r 
    
    E_u = - H - J * sl_s
    E_d =   H + J * sl_s 
    
    return E_u, E_d

def probabilities_spins(i_list, j_list, sl, H, J, T, kb):
    
    E_u, E_d = energies_spins(i_list, j_list, sl, H, J)
    
    Ei = np.array([E_u, E_d])
    
    Z = np.sum(np.exp(- Ei / (kb*T)), axis=0)
    pi = 1 / np.array([Z, Z]) * np.exp(- Ei / (kb*T))

    return pi, Z   

def IsingModelStep(sl, H, J, T, kb, Nspins, S):
    Ni, Nj = sl.shape

    ns = np.random.choice(range(Nspins), S)

    i_list = list(map(lambda x: x % Ni, ns))
    j_list = list(map(lambda x: x // Ni, ns))

    pi, Z = probabilities_spins(i_list, j_list, sl, H, J, T, kb)

    rn = np.random.rand(S)
    for i in range(S):
        if rn[i] > pi[0, i]:
            sl[i_list[i], j_list[i]] = -1
        else:
            sl[i_list[i], j_list[i]] = 1
    
    return sl

#%% Parameters

N = 100     # Bigger if possible
J = 1
kb = 1

#%% Task 1
H_list = [-5, -2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 5]
T = 5

max_iterations = 1000

sl = 2 * np.random.randint(2, size=(N,N)) - 1

Nspins = np.size(sl)
S = int(np.ceil(Nspins * 0.05))

m_array = []

for H in H_list:
    print(f'Simulating for H = {H}')
    m_temp = []

    for iteration in range(max_iterations):
        sl = IsingModelStep(sl, H, J, T, kb, Nspins, S)

        if iteration > max_iterations - 200:
            m_temp.append(1/Nspins * np.sum(sl))

    m_array.append(np.mean(m_temp))

# Plot
p = np.polyfit(H_list[4:9], m_array[4:9], 1)
chi = p[0]

plt.plot(H_list, m_array, '.-')
plt.plot(H_list, p[0]*np.array(H_list) + p[1], '--k')
plt.ylim([-1,1])
plt.xlabel(r'$H$')
plt.ylabel(r'$m(H)$')
plt.title(r'Magnetization $m(H)$ as a function of the magnetic field $H$')
plt.grid()
plt.show()

print(f'Calculated value of chi: {chi}')

#%% Task 2
T_list = [0.1, 0.2, 0.5, 1, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 5]
H = 0.1

max_iterations = 5000

sl = 2 * np.random.randint(2, size=(N,N)) - 1

Nspins = np.size(sl)
S = int(np.ceil(Nspins * 0.05))

m_array = []

for T in T_list:
    print(f'Simulating for T = {T}')
    m_temp = []

    for iteration in range(max_iterations):
        if iteration > 300:
            H = 0

        sl = IsingModelStep(sl, H, J, T, kb, Nspins, S)

    m_array.append(1/Nspins * np.sum(sl))

#%% Plot
plt.plot(T_list, m_array, '.-')
plt.xlabel(r'$T$')
plt.ylabel(r'$m(T)$')
plt.title(r'Magnetization $m(T)$ as a function of the temperature $T$')
plt.grid()
plt.show()
# %%
