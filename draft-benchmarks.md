# draft-benchmarks

## other cite

​	首先，观察其他类似文章的benchmark：

​	LiuAn的一些列工作：

[^RobustRecovery]: A. Liu, G. Liu, L. Lian, V. K. N. Lau and M. Zhao, "Robust Recovery of Structured Sparse Signals With Uncertain Sensing Matrix: A Turbo-VBI Approach," in IEEE Transactions on Wireless Communications, vol. 19, no. 5, pp. 3185-3198, May 2020, doi: 10.1109/TWC.2020.2971193.
[^AngularDomain]: G. Liu, A. Liu, R. Zhang and M. Zhao, "Angular-Domain Selective Channel Tracking and Doppler Compensation for High-Mobility mmWave Massive MIMO," in IEEE Transactions on Wireless Communications, vol. 20, no. 5, pp. 2902-2916, May 2021, doi: 10.1109/TWC.2020.3045272.
[^Downlink]: A. Liu, L. Lian, V. K. N. Lau and X. Yuan, "Downlink Channel Estimation in Multiuser Massive MIMO With Hidden Markovian Sparsity," in IEEE Transactions on Signal Processing, vol. 66, no. 18, pp. 4796-4810, 15 Sept.15, 2018, doi: 10.1109/TSP.2018.2862420. 
[^DynamicSparsity]: L. Lian, A. Liu and V. K. N. Lau, "Exploiting Dynamic Sparsity for Downlink FDD-Massive MIMO Channel Tracking," in IEEE Transactions on Signal Processing, vol. 67, no. 8, pp. 2007-2021, 15 April15, 2019, doi: 10.1109/TSP.2019.2896179.
[^CloudAssisted]: A. Liu, L. Lian, V. Lau, G. Liu and M. Zhao, "Cloud-Assisted Cooperative Localization for Vehicle Platoons: A Turbo Approach," in IEEE Transactions on Signal Processing, vol. 68, pp. 605-620, 2020, doi: 10.1109/TSP.2020.2964198.
[^FDD]: J. Dai, A. Liu and V. K. N. Lau, "FDD Massive MIMO Channel Estimation With Arbitrary 2D-Array Geometry," in IEEE Transactions on Signal Processing, vol. 66, no. 10, pp. 2584-2599, 15 May15, 2018, doi: 10.1109/TSP.2018.2807390.

​	PanCunhua 的一些工作：

[^PanCunhua]:  Zhou, Gui, et al. "Channel estimation for RIS-aided multiuser millimeter-wave systems." *arXiv preprint arXiv:2106.14792* (2021).



## Performance and indicators

首先确定实验结果需要什么：



### convergence 1

MM method 的有效性

### convergence 2

相比其他算法的快速收敛



### NMSE-SNR

所提出的算法：Structure Sparsity boost Hybrid EM

选择benchmark的目的：

1. 对比传统方法NMSE更低：AMP、OMP、LASSO、PanCunhua-OMP
2. 相对于Turbo-VBI框架有创新：non-Common Hybrid EM

### NMSE-tau

