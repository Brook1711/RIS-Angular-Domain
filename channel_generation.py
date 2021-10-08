# %%
import numpy as np
import math, cmath
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
# system parameters
distance_rb = 20    # the distance of the RIS-BS or BS-RIS link
sigma_rb = -2.2     # the fading factor of the BS-RIS link
sigma_rk = -2.8     # the fading factor fo the RIS-user link
L = 4               # the number of the multi-paths in BS-RIS link


# %%

def generate_A(distance, L=4, sigma_rb=-2.2):
    path_loss = 10e-3*np.power(distance, sigma_rb)
    A = np.diag(
        [a[0] for a in np.random.randn(L,2).view(np.complex128)*np.power(2,-1/2)]
        )*np.power(path_loss, 1/2)
    return np.mat(A), path_loss

def generate_B_k(distance, J_k=4, sigma_rk=-2.8):
    path_loss = 10e-3*np.power(distance, sigma_rk)
    B = np.diag(
        [a[0] for a in np.random.randn(J_k,2).view(np.complex128)*np.power(2,-1/2)]
        )*np.power(path_loss, 1/2)
    return np.mat(B), path_loss

# %%
np.random.seed(1234)
A = generate_A()