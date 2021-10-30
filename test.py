# 首先定义一组角度是从0到2pi，同时论文中的角度是-0.5到0.5，中间有一个映射关系
# 方便起见，我们在定义角度的时候直接定义为-pi到pi，即，角度和论文里的变量存在pi的线性倍数关系

import math, cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 定义函数D_N
def D_N(N, x):
    
    return cmath.exp(1j*x*(N+1)/2)*(cmath.sin(N*x/2))/(cmath.sin(x/2))


M = 16
x = np.array(range(M))
varphi_list = [-0.3,0.3]
varphi_all_grid = [(m+1 -1)/M - 0.5 for m in range(M)]
varphi_info = [{"gridIndex":0,"varphiGrid":0,"offGrid":0} for i in range(len(varphi_list))]

for i, varphi in enumerate(varphi_list):
    for j, varphiGrid in enumerate(varphi_all_grid):
        if abs(varphiGrid - varphi) <= 0.5 / M:
            varphi_info[i]["gridIndex"] = j
            varphi_info[i]["varphiGrid"] = varphiGrid
            varphi_info[i]["offGrid"] = varphiGrid - varphi
print(varphi_info)

varphi_support_array = [1 if m in [varphi["gridIndex"] for varphi in varphi_info] else 0 for m in range(M)]
print(varphi_support_array)

arrayres_M_1 = np.exp(1j*2*np.pi*x*varphi_list[0])/np.power(M,1/2)
arrayres_M_2 = np.exp(1j*2*np.pi*x*varphi_list[1])/np.power(M,1/2)

sum_array = arrayres_M_1 + arrayres_M_2
dftmtx = np.mat(np.fft.fft(np.eye(M)))

plt.plot(x, np.abs(np.array(dftmtx * np.reshape(np.mat(sum_array),(M,-1)))))
# plt.plot(x, 1+np.abs(np.array(dftmtx * np.reshape(np.mat(sum_array),(M,-1)))))


