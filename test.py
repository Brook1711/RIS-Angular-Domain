# 首先定义一组角度是从0到2pi，同时论文中的角度是-0.5到0.5，中间有一个映射关系
# 方便起见，我们在定义角度的时候直接定义为-pi到pi，即，角度和论文里的变量存在2pi的线性倍数关系

import math, cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 定义函数D_N
def D_N(N, x):
    
    return cmath.exp(1j*x*(N+1)/2)*(cmath.sin(N*x/2))/(cmath.sin(x/2))


M = 16
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

varphi_support_array = [1 if m in [varphi["gridIndex"] for varphi in varphi_info] else 0 for m in range(M)]
print(varphi_support_array)

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
# print((D_N_1+D_N_2)[0] / np.abs(np.array(dftmtx * np.reshape(np.mat(sum_array),(M,-1))))[0])
# print((D_N_1+D_N_2)[1] / np.abs(np.array(dftmtx * np.reshape(np.mat(sum_array),(M,-1))))[1])
# plt.figure(2)
# D_N_1 = np.array([abs(D_N(M, 2*cmath.pi * (m - M*0.1)/M )) for m in range(M)])


plt.scatter(x, (abs(D_N_3)) )
# plt.scatter(x, (abs(D_N_3)) )

plt.scatter(x, np.abs(np.array(dftmtx * np.reshape(np.mat(arrayres_M_2),(M,-1)))))
plt.figure(2)
DFT_add = np.abs(np.array(
    dftmtx * np.reshape(np.mat(arrayres_M_2),(M,-1)) + dftmtx * np.reshape(np.mat(arrayres_M_1),(M,-1))
    ))
add_DFT = np.abs(np.array(
    dftmtx * (np.reshape(np.mat(arrayres_M_2),(M,-1)) + np.reshape(np.mat(arrayres_M_1),(M,-1)))
    ))
plt.scatter(x, DFT_add)
plt.scatter(x, add_DFT *0.5)

plt.figure(3)
plt.scatter(x, (abs(D_N_1)) )
plt.scatter(x, 4*abs(np.array(DFT2)))
