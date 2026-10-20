# 14-OCT
# Sandbox file for testing SEIR-model constructions

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp

from scipy.integrate import odeint

pop_size = 3.3e8

gamma = 1 / 18
sigma = 1 / 5.2

def F(x, t, R0=1.6):
    """
    Time derivative of the state vector.

        * x is the state vector (array_like)
        * t is time (scalar)
        * R0 is the effective transmission rate, defaulting to a constant

    """
    s, e, i = x

    # New exposure of susceptibles
    beta = R0(t) * gamma if callable(R0) else R0 * gamma
    ne = beta * s * i

    # Time derivatives
    ds = - ne
    de = ne - sigma * e
    di = sigma * e - gamma * i

    return ds, de, di

i_0 = 1e-7
e_0 = 4 * i_0
s_0 = 1 - i_0 - e_0

x_0 = s_0, e_0, i_0
def solve_path(R0, t_vec, x_init=x_0):
    """
    Solve for i(t) and c(t) via numerical integration,
    given the time path for R0.

    """
    G = lambda x, t: F(x, t, R0)
    s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path       # cumulative cases
    return i_path, c_path

t_length = 550
grid_size = 1000
t_vec = np.linspace(0, t_length, grid_size)

R0_vals = np.linspace(1.6, 3.0, 6)
labels = [f'$R_0 = {r:.2f}$' for r in R0_vals]
i_paths, c_paths = [], []

for r in R0_vals:
    i_path, c_path = solve_path(r, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)

def plot_paths(paths, labels, ylabel, times=t_vec, mainlabel=''):

    fig, ax = plt.subplots()

    for path, label in zip(paths, labels):
        ax.plot(times, path, lw=2, label=label)

    ax.set_xlabel('days')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    plt.title(mainlabel)

    plt.show()

plot_paths(i_paths, labels, ylabel='Fraction of the population',
           mainlabel='Infections over time')

plot_paths(c_paths, labels, ylabel='Fraction of the population',
           mainlabel='Cumulative cases over time')



# Example of using count data to infer the transmission rate


