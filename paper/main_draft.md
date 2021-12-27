# Title: Turbo-VBI-based Channel Acquization in Single-Antanna assisted RIS system



# Abstract:

提出一种新的硬件架构：单天线辅助的可重构智能表面（Single-Antanna assisted Reconfigurable Intelligent Surface, SA-RIS）。

设计了针对SA-RIS的信道信息获取协议。

采用角域信道建模，对协议各个阶段进行分析。

提出了一种适用于SA-RIS系统中基于VBI和MP的低复杂度信道估计算法。



# Part 1: Introduction:

> 介绍背景
>
> 1. RIS发展背景
> 2. 传统RIS中的信道估计问题



> 介绍RIS信道估计
>
> 1. active RIS信道估计
>    1. 优点：估计精度高，协议设计简单
>    2. 缺点：需要在RIS上安装大量射频链路，与RIS的设计初衷违背
> 2. passive RIS信道估计
>    1. 优点：硬件结构简单
>    2. 缺点：算法设计复杂，计算复杂度较高，估计精度较差



针对上述两种RIS架构，我们提出了一种综合两种优势的SA-RIS架构

通过单天线广播，在开始时可以获知很多级联信道中的信道信息[^AngularEstimation]，



> 介绍算法背景
>
> 1. 传统算法
>    1. 单纯的MP[^pure-MP][^pure-MP-2]
>    2. OMP[^PanCunhua]
> 2. 新型算法(还没有在RIS系统中得到应用)
>    1. T-VBI
>       1. ChannelTracking-[^AngularDomain] [^RobustRecovery][^DynamicSparsity]
>    2. Factor Graph
>       1. basic[^FactorGraph]
>    3. Three layer HMM隐马尔可夫模型
>       1. proposed[^RobustRecovery]



为了达到更高的估计精度，我们将角域中的AoA/AoD基于Grid的角偏移量通过DFT公示纳入考虑[^SpectralCS]

## Contribution:

1. 提出一种新的架构：SA-RIS
2. 针对SA-RIS提出适合的信道估计协议
3. 针对级联信道估计问题设计SA-RIS中的低复杂度信道估计算法



# Part 2: SYSTEM MODEL AND PROBLEM FORMULATION



## A. SA-RIS-Assisted Multiuser MIMO system

考虑一个RIS辅助的多用户上行毫米波MIMO通信系统，如图一所示。该系统中BS装备有一个规模为$N \times 1$的ULA天线阵列。RIS装备有规模为$M\times 1$的UPA反射阵列。服务用户均为装备有$A$个天线的ULA接收端。



[^pure-MP]: H. Liu, X. Yuan and Y. -J. A. Zhang, "Matrix-Calibration-Based Cascaded Channel Estimation for Reconfigurable Intelligent Surface Assisted Multiuser MIMO," in *IEEE Journal on Selected Areas in Communications*, vol. 38, no. 11, pp. 2621-2636, Nov. 2020, doi: 10.1109/JSAC.2020.3007057.
[^pure-MP-2]: H. Liu, X. Yuan and Y. -J. A. Zhang, "Message-Passing Based Channel Estimation for Reconfigurable Intelligent Surface Assisted MIMO," *2020 IEEE International Symposium on Information Theory (ISIT)*, 2020, pp. 2983-2988, doi: 10.1109/ISIT44484.2020.9173987.
[^PanCunhua]:  Zhou, Gui, et al. "Channel estimation for RIS-aided multiuser millimeter-wave systems." *arXiv preprint arXiv:2106.14792* (2021).
[^AngularDomain]: G. Liu, A. Liu, R. Zhang and M. Zhao, "Angular-Domain Selective Channel Tracking and Doppler Compensation for High-Mobility mmWave Massive MIMO," in IEEE Transactions on Wireless Communications, vol. 20, no. 5, pp. 2902-2916, May 2021, doi: 10.1109/TWC.2020.3045272.
[^FactorGraph]: F. R. Kschischang, B. J. Frey and H. -. Loeliger, "Factor graphs and the sum-product algorithm," in *IEEE Transactions on Information Theory*, vol. 47, no. 2, pp. 498-519, Feb 2001, doi: 10.1109/18.910572.
[^RobustRecovery]: A. Liu, G. Liu, L. Lian, V. K. N. Lau and M. Zhao, "Robust Recovery of Structured Sparse Signals With Uncertain Sensing Matrix: A Turbo-VBI Approach," in IEEE Transactions on Wireless Communications, vol. 19, no. 5, pp. 3185-3198, May 2020, doi: 10.1109/TWC.2020.2971193.
[^DynamicSparsity]: L. Lian, A. Liu and V. K. N. Lau, "Exploiting Dynamic Sparsity for Downlink FDD-Massive MIMO Channel Tracking," in IEEE Transactions on Signal Processing, vol. 67, no. 8, pp. 2007-2021, 15 April15, 2019, doi: 10.1109/TSP.2019.2896179.
[^SpectralCS]: Duarte, Marco F., and Richard G. Baraniuk. "Spectral compressive sensing." *Applied and Computational Harmonic Analysis* 35.1 (2013): 111-129.
[^AngularEstimation]: P. Zhao, K. Ma, Z. Wang and S. Chen, "Virtual Angular-Domain Channel Estimation for FDD Based Massive MIMO Systems With Partial Orthogonal Pilot Design," in IEEE Transactions on Vehicular Technology, vol. 69, no. 5, pp. 5164-5178, May 2020, doi: 10.1109/TVT.2020.2979916.



