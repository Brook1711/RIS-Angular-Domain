# %%
import numpy as np
import math, cmath
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy.io import savemat, loadmat
# %%
# system parameters
class channel_parameters(object):
    cp_dic = {}
    cp_dic['N']=64
    cp_dic['M']=64
    cp_dic['A']=32

    cp_dic['distance_rb'] = 20    # the distance of the RIS-BS or BS-RIS link
    cp_dic['distance_rk'] = 10    # the distance of the RIS-BS or user-RIS link

    cp_dic['sigma_rb'] = -2.2     # the fading factor of the BS-RIS link
    cp_dic['sigma_rk'] = -2.8     # the fading factor fo the RIS-user link
    cp_dic['L'] = 4               # the number of the multi-paths in BS-RIS link
    cp_dic['J_k'] = 4             # the number of the multi-paths in RIS-user link 

    cp_dic['d_div_lambda_bs'] = 0.5
    cp_dic['d_div_lambda_ris'] = 0.5
    cp_dic['d_div_lambda_user'] = 0.5

    def __init__(self, cp_new_dic = None) -> None:
        if cp_new_dic:
            for key in cp_new_dic.keys():
                if key in self.cp_dic.keys():
                    self.cp_dic[key] = cp_new_dic[key]
        pass

    
cp_0 = channel_parameters()
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

def generate_angle(dimension):
    # return a 'dimension' dimensions array that uniform randomly form -pi/2 ~pi/2
    angle_array = np.random.uniform(-1, 1, dimension)*np.pi/2

    return angle_array

# %%
# generating the complex path loss diag matrix of the BS-RIS link
np.random.seed(1234)
A,A_pl = generate_A(
    distance=cp_0.cp_dic['distance_rb'], 
    L=cp_0.cp_dic['L'], 
    sigma_rb= cp_0.cp_dic['sigma_rb']
    )

# %%
# generating the complex path loss diag matrix of the user-RIS link
np.random.seed(4321)
B_k, B_k_pl = generate_B_k(
    distance=cp_0.cp_dic['distance_rk'], 
    J_k=cp_0.cp_dic['J_k'], 
    sigma_rk= cp_0.cp_dic['sigma_rk']
    )
# %%
# generate angles (4 groups psi, omega, varphi, phi)

np.random.seed(1234)
psi = generate_angle(dimension=cp_0.cp_dic['L'])
psi_prime = cp_0.cp_dic['d_div_lambda_bs'] * np.cos(psi)
# generate A_N


np.random.seed(2341)
omega = generate_angle(dimension=cp_0.cp_dic['L'])
omega_prime = cp_0.cp_dic['d_div_lambda_ris'] * np.cos(omega)
# generate A_M

np.random.seed(3412)
varphi = generate_angle(dimension=cp_0.cp_dic['J_k'])
varphi_prime = cp_0.cp_dic['d_div_lambda_ris'] * np.cos(varphi)
# generate A_Mk

np.random.seed(4123)
phi = generate_angle(dimension=cp_0.cp_dic['J_k'])
phi_prime = cp_0.cp_dic['d_div_lambda_user'] * np.cos(phi)
# generate A_Ak

# %%
# store data to mat file 
dic_store = {}

dic_store['channel_parameters']=cp_0.cp_dic

dic_store['A']=A
dic_store['B_k']=B_k

dic_store['psi'] = psi
dic_store['psi_prime'] = psi_prime

dic_store['omega'] = omega
dic_store['omega_prime'] = omega_prime

dic_store['varphi'] = phi
dic_store['varphi_prime'] = varphi_prime

dic_store['phi'] = phi
dic_store['phi_prime'] = phi_prime

savemat('build/data/test_save.mat', dic_store)
np.save('build/data/channel_data.npy',dic_store,allow_pickle=True)

# %%
# test load data
a = loadmat('build/data/test_save.mat')
np_load = np.load('build/data/channel_data.npy',allow_pickle=True).item()
print("========================channel generation==========================")
# %%
