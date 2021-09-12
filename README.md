# RIS-Angular-Domain

主要参考文章[^1]，首先复现该文章的角度域解法，同时对比其他benchmark

## 研究背景：

高速移动场景下的多普勒效应和多经效应难以解决，文了解决这一挑战，文中列举了四种研究方向：

- **直接对信道进行估计或预测（direct channel estimation/ prediction）**
  - 线性时变信道模型（Linearly Time-Varying, LTV）
  - 基础扩展模型（basis expansion model, BEM）
- **正交时间空间频率调制（Orthogonal Time Frequency Space(OTFS) Modulation）**
  - 将时变多经信道转化为（time-invariant channel）时不变信道
- **角度域的DFO估计和补偿**
  - 基本思想：考虑到DFO的产生本身就是多径中AoA和AoD的不同造成的，所以直接从角度域进行DFO的估计和补偿。目前，在小规模和大规模MIMO系统中均有角度域估计的研究，但是基于最大似然（ML）的MIMO信道联合估计方法会引入较大的信道开销；而且无法获得空分增益和阵列信噪比增益。
- **基于码本的波束赋形和波束跟踪**
  - 不通过信道估计，直接尝试现有码本中的方案
  - 遍历搜索（Exhaustive Search, ES）方案
  - 分层搜索（hierarchical search，HS）方案。ES的低开销改进版
  - 问题：量化误差（quantization error）和信道老化（channel aging）；针对快衰落信道效果不佳（快速移动场景效果不好）

本文中发端和收端均采用大规模MIMO天线，在提高谱效的同时提高高速移动场景下的链路可靠性。

但是由于高维度的信道矩阵会造成高信道开销，并且现有方法（基于压缩感知 Compressive Sensing，CS ）主要应用在满衰落信道中，无法应用在快衰落的高速移动场景下。

所以提出了一种方法：

- 角度域选择性信道跟踪和多普勒效应补偿方法（Angular-domain selective channel tracking and Doppler compensation scheme）
  - 利用了mmWave大规模MIMO信道的动态稀疏性（dynamic sparsity）
  - 在上下行链路应用预编码训练（precoded training）

主要贡献：

- 角度域选择性信道跟踪
- 角度域选择性信道补偿
- 基于动态变分bayesian 接口（dynamic VBI）
  - three layer hierarchical Markov model
  - 变分贝叶斯推理（Variational Bayesian Inference, VBI）;稀疏贝叶斯学习（Sparse Bayesian Learning, SBL）无法直接应用
  - 提出新的多普勒感知的动态VBI（Doppler-aware-dynamic Variational Bayesian inference, DD-VBI）
    - 该方法将VBI和信息传递方法（message-passing approaches）相结合

## 系统模型：

单用户时分系统

分为上行和下行两种subframe

每个$t$-th subframe中都有$N_p$个相同的训练向量$\bf{v}_t$，这保证了在用户端可以估计（部分的）信道特征。

然后基于估计的特征，用户会使用多普勒补偿矩阵（Doppler compensation matrix）对多普勒效应作出补偿。在本质上将快速时变信道转换为慢时变有效信道（slow time-arying effective channel）


$$
\begin{array}{|c|c|c|c|}
\hline \text { Notation } & \text { Meaning } & \text { Notation } & \text { Meaning } \\
\hline \hline N_{p} & \text { Number of downlink training vectors } & \theta_{T, t, q}\left(\theta_{R, t, q}\right) & \text { The AoD (AoA) of the } q \text {-th path } \\
\hline \mathbf{v}_{t} & \text { Downlink training vector } & \eta_{t} & \text { Rotation angle of user's antenna array } \\
\hline M(N) & \text { Number of BS (user) antennas } & \tilde{\theta}_{T, m}\left(\tilde{\theta}_{R, m}\right) & m \text {-th AoD grid (AoA grid) } \\
\hline L_{t} & \text { Number of propagation paths } & \boldsymbol{\beta}_{T, t}\left(\boldsymbol{\beta}_{R, t}\right) & \text { The AoD(AoA) off-grid vector } \\
\hline \alpha_{t, q} & \text { The path gain of the } q \text {-th path } & N_{b} & \text { Number of RF chains at the user } \\
\hline f_{d, t} & \text { The maximum DFO } & \tilde{M}(\tilde{N}) & \text { Number of AoD (AoA) grid } \\
\hline
\end{array}
$$

[^1]: G. Liu, A. Liu, R. Zhang and M. Zhao, "Angular-Domain Selective Channel Tracking and Doppler Compensation for High-Mobility mmWave Massive MIMO," in IEEE Transactions on Wireless Communications, vol. 20, no. 5, pp. 2902-2916, May 2021, doi: 10.1109/TWC.2020.3045272.
[^]: 









>  # Noted
>
> 绘图采用liboffice绘制

