# %%
import numpy as np
import math, cmath
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
# system parameters
distance_rb = 20    # the distance of the RIS-BS or BS-RIS link
distance_rk = 10    # the distance of the RIS-BS or user-RIS link

sigma_rb = -2.2     # the fading factor of the BS-RIS link
sigma_rk = -2.8     # the fading factor fo the RIS-user link
L = 4               # the number of the multi-paths in BS-RIS link
J_k = 4             # the number of the multi-paths in RIS-user link 

# %%

def ULA_array_response(X, theta, d, lambda_c):
    # Input:
        # X: the dimension of the array, int
        # theta: the AoA/AoD of the in/out signal, float(0~$\pi$)
        # d: array interval
        # lambda_c: carrier wave length

    # Output:
        # alpha: the array response
    alpha = np.array(X*[0+1j])
    for x in range(X):
        phase = 2*math.pi*x*(d/lambda_c)*math.cos(theta)
        alpha[x] = cmath.exp(-1j * phase)
    return np.mat(alpha).T

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
# generating the complex path loss diag matrix of the BS-RIS link
np.random.seed(1234)
A = generate_A(
    distance=distance_rb, 
    L=L, 
    sigma_rb= sigma_rb
    )
print(A)

# %%
# generating the complex path loss diag matrix of the user-RIS link
np.random.seed(4321)
B_k = generate_B_k(
    distance=distance_rk, 
    J_k=J_k, 
    sigma_rk= sigma_rk
    )
print(B_k)
# %%
