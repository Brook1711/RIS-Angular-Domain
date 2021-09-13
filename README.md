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

### 帧结构

![image-20210913104714230](README.assets/image-20210913104714230.png)

单用户时分系统

分为上行和下行两种subframe

每个$t$-th subframe中都有$N_p$个相同的训练向量$\bf{v}_t$，这保证了在用户端可以估计（部分的）信道特征。

- 在下行链路中，基于估计的特征，用户会使用多普勒补偿矩阵（Doppler compensation matrix）对多普勒效应作出补偿。在本质上**将快速时变信道转换为慢时变有效信道**（slow time-arying effective channel）
- 在上行链路中，在每个subframe的头和尾有两种训练向量，该向量是用于在多普勒补偿之后估计慢时变有效通道（slow time-varying effective channel）

上行链路在$t$-thsubframe中优化

下行链路在$(t-1)$-th subframe后优化

### 多普勒和多径信道模型

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



基站和用户都使用半波长间隔ULA

由于系统工作在窄带，所以信道衰落为平稳衰落，用下式表示在$t-th$subframe中$i-th$ symbol的下行信道矩阵[^1-21]：
$$
\boldsymbol{H}_{t, i}=\sum_{q=1}^{L_{t}} \alpha_{t, q} e^{j 2 \pi f_{d, t} i \cos \left(\theta_{R, t, q}+\eta_{t}\right)} \boldsymbol{a}_{R}\left(\theta_{R, t, q}\right) \boldsymbol{a}_{T}^{H}\left(\theta_{T, t, q}\right)
$$

$f_{d,t}$ 泛化最大DFO

$\eta_{t}$ 用户相对于运动方向的`rotation angle`

### 角度域信道表示

由于假设每个subframe内当前symbol所有参数恒定不变，则在后文的表达式中省略所有变量中的$t$ 脚标

接下来分别对AoD和AoD在$
[-\pi / 2, \pi / 2]$上进行$\tilde{M}$和$\tilde{N}$离散化

但是离散化必然会导致量化误差，所以本文提出了[^1-24]

>  off-grid basis for the angular domain channel representation

令$\tilde{\theta}_{T, m_{q}}$和$\tilde{\theta}_{R, n_{q}}$表示距离真实角度$\theta_{T, q}$和$\theta_{R, q}$最近的离散角度，引入`off-grid vector`的概念：$\boldsymbol{\beta}_{T}=\left[\beta_{T, 1}, \beta_{T, 2}, \ldots, \beta_{T, \tilde{M}}\right]^{T}$该向量满足：

$\beta_{T, m}= \begin{cases}\theta_{T, q}-\tilde{\theta}_{T, m_{q}}, & m=m_{q}, \quad q=1,2, \ldots, L \\ 0, & \text { otherwise }\end{cases}$

$\beta_{R, n}= \begin{cases}\theta_{R, q}-\tilde{\theta}_{R, n_{q}}, & n=n_{q}, \quad q=1,2, \ldots, L \\ 0, & \text { otherwise }\end{cases}$

> # Noted
>
> 本质上来讲off-grid vector 是所有路径的量化误差向量





[^1]: G. Liu, A. Liu, R. Zhang and M. Zhao, "Angular-Domain Selective Channel Tracking and Doppler Compensation for High-Mobility mmWave Massive MIMO," in IEEE Transactions on Wireless Communications, vol. 20, no. 5, pp. 2902-2916, May 2021, doi: 10.1109/TWC.2020.3045272.
[^1-21]: [W. U. Bajwa, J. Haupt, A. M. Sayeed, and R. Nowak, “Compressed channel sensing: A new approach to estimating sparse multipath channels,” Proc. IEEE, vol. 98, no. 6, pp. 1058–1076, Jun. 2010.](doc/1-21_channel.md)
[^1-24]: J. Dai, A. Liu, and V. K. N. Lau, “FDD massive MIMO channel estimation with arbitrary 2D-array geometry,” IEEE Trans. Signal Process., vol. 66, no. 10, pp. 2584–2599, May 2018.









>  # Noted
>
> 绘图采用liboffice绘制

