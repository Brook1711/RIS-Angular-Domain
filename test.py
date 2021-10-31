# 首先定义一组角度是从0到2pi，同时论文中的角度是-0.5到0.5，中间有一个映射关系
# 方便起见，我们在定义角度的时候直接定义为-pi到pi，即，角度和论文里的变量存在2pi的线性倍数关系

import math, cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 定义函数D_N
def D_N(N, x):
    if abs(x) == 0:
        return N
    else:
        return cmath.exp(1j*x*(N-1)/2)*(cmath.sin((N)*x/2))/(cmath.sin(x/2))


M = 16
sqrt_M = np.sqrt(M)
x = np.array(range(M))
varphi_list = [-0.1,0.1]
varphi_all_grid = [(m+1 -1)/M - 0.5 for m in range(M)]
varphi_info = [{"gridIndex":0,"varphiGrid":0,"offGrid":0} for i in range(len(varphi_list))]

for i, varphi in enumerate(varphi_list):
    for j, varphiGrid in enumerate(varphi_all_grid):
        if abs(varphiGrid - varphi) <= 0.5 / M:
            if varphiGrid >= 0:
                varphi_info[i]["gridIndex"] = M * varphiGrid 
            else:
                varphi_info[i]["gridIndex"] = M * varphiGrid + M
            varphi_info[i]["varphiGrid"] = varphiGrid
            varphi_info[i]["offGrid"] = varphiGrid - varphi
print(varphi_info)

varphi_support_array = np.array([1 if m in [varphi["gridIndex"] for varphi in varphi_info] else 0 for m in range(M)])
print(varphi_support_array)
varphi_support_matrix = np.mat(varphi_support_array.reshape((M,1)))
arrayres_M_1 = np.exp(1j*2*np.pi*x*varphi_list[0])/np.power(M,1/2)
arrayres_M_2 = np.exp(1j*2*np.pi*x*varphi_list[1])/np.power(M,1/2)


sum_array = arrayres_M_1 + arrayres_M_2
dftmtx = np.mat(np.fft.fft(np.eye(M)))
DFT1 = dftmtx * np.reshape(np.mat(arrayres_M_1),(M,-1))
DFT2 = dftmtx * np.reshape(np.mat(arrayres_M_2),(M,-1))

# plt.scatter(x, np.abs(np.array(dftmtx * np.reshape(np.mat(sum_array),(M,-1)))))
# plt.plot(x, 1+np.abs(np.array(dftmtx * np.reshape(np.mat(sum_array),(M,-1)))))

# D矩阵的第m列：
# [D_N(M, 2*cmath.pi/M * (m+1 -M/(2*cmath.pi)*(varphiGrid)  )) for m in range(M)]
D_N_1 = np.array([(D_N(M, 2*cmath.pi *(m/M - 2/M + 0.025) )) for m in range(M)])
D_N_2 = np.array([(D_N(M, 2*cmath.pi *(m/M - 14/M + 1 - 0.025) )) for m in range(M)])
D_N_3 = np.array([D_N(M, 2*cmath.pi *(m/M - 2/M + 0.025) ) +D_N(M, 2*cmath.pi *(m/M - 14/M + 1 - 0.025) ) for m in range(M)])

Delta_varphi = [0 for m in range(M)]
for varphi_info_i in varphi_info:
    Delta_varphi[int(varphi_info_i['gridIndex'])] = varphi_info_i['offGrid']

def generate_DM(Delta_varphi):
    M_ = len(Delta_varphi)
    DM = np.matrix([np.complex128(0) for m in range(M_ * M_)]).reshape((M_,M_))
    for col in range(M_):
        if col/M <0.5:
            DM[:,col] = np.array([(D_N(M_, 2*cmath.pi *(m/M_ - col/M_ + Delta_varphi[col]) )) for m in range(M_)]).reshape(M_,1)
        else:
            DM[:,col] = np.array([(D_N(M_, 2*cmath.pi *(m/M_ - col/M_ + 1 + Delta_varphi[col]) )) for m in range(M_)]).reshape(M_,1)
    return DM

D_M = generate_DM(Delta_varphi)
plt.scatter(x, (abs(D_N_3)) )
plt.scatter(x, 4*np.abs(np.array(dftmtx * np.reshape(np.mat(arrayres_M_2+arrayres_M_1),(M,-1)))))
plt.scatter(x, np.abs(np.array(D_M * varphi_support_matrix)))
