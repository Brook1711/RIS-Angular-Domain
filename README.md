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
> $q$和$m_q$是一一对应的关系，这个关系在刚开始制定量化划分的时候就需要足够密集以保证每个$q$可以分得一个唯一的grid point

> # Noted
>
> 本质上来讲off-grid vector 是所有路径的量化误差向量，但并不代表误差，他其实代表了一种定位手段。在这种定位手段假设量化的所有grid point上均有一个对应的传播路径，至于grid point和实际的误差和grid point 上到底有没有传播路径，则是由$\bf{\beta}_T$和${\bf \beta}_R$所描述的

$$
\boldsymbol{A}_{R, i}(\boldsymbol{\varphi})=\left[\tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{1}\right), \ldots, \tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{N}\right)\right] \in \mathbb{C}^{N \times N}
$$

$$
\boldsymbol{A}_{T}\left(\boldsymbol{\beta}_{T}\right)=\left[\boldsymbol{a}_{T}\left(\tilde{\theta}_{T, 1}+\beta_{T, 1}\right), \ldots, \boldsymbol{a}_{T}\left(\tilde{\theta}_{T, \bar{M}}+\beta_{T, \bar{M}}\right)\right] \in \mathbb{C}^{M \times \bar{M}}
$$

$$
\tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{n}\right)=\boldsymbol{a}_{R}\left(\tilde{\theta}_{R, n}+\beta_{R, n}\right) \times e^{j 2 \pi f_{d} i \cos \left(\tilde{\theta}_{R, n}+\beta_{R, n}+\eta\right)}
$$

为了可以写成更加紧凑的矩阵形式，定义了矩阵$\tilde{X}$表示对应path的path loss：
$$
\tilde{x}_{n, m}= \begin{cases}\alpha_{q}, & (n, m)=\left(n_{q}, m_{q}\right), \quad q=1,2, \ldots, L \\ 0, & \text { otherwise. }\end{cases}
$$
最后的MIMO信道矩阵可以表示为：
$$
\boldsymbol{H}_{i}\left(\boldsymbol{\varphi}, \boldsymbol{\beta}_{T}\right)=\boldsymbol{A}_{R, i}(\boldsymbol{\varphi}) \tilde{\boldsymbol{X}} \boldsymbol{A}_{T}^{H}\left(\boldsymbol{\beta}_{T}\right)
$$
注意到，以上推导也可以在二维天线阵列中进行推导

> ##  个人理解
>
> 以上步骤只是将多径求和换了一种写法，在量化之后就又通过${\bf \beta}_T$和${\bf \beta}_R$ 弥补了量化误差，实际结果和直接写求和是一样的

### 角度域选择性信道跟踪和多普勒补偿

- mmWave massive MIMO 信道的动态稀疏性（dynamic sparsity）
- 利用用户端多天线阵列的高AoA分辨率

来估计：信道参数、AoA、rotation angle、maximum DFO

- 目的：
  - 将高维快速衰落信道转化为低维满衰落信道

- 关键技术：
  - 角度域选择性信道跟踪
  - 选择性多普勒补偿
  - 满衰落信道估计
  - 下行训练向量设计

> ## Star
>
> 注意以下和索引$t$ 基本无关，将其省略



#### A 用户端的角度域选择性信道跟踪

- 目的
  - 估计用于多普勒补偿的信道特征

由于信道模型中的DFO参数和AoA存在一对一的对应关系，而且用户处的大规模天线带来的空间高分辨率可以将不同DFO从不同AoA中分离出来

但是这样的代价是由于BS和用户都使用大规模天线，使得参数维度过高，需要估计的参数有：全角度域信道矩阵${\tilde{X}}$、rotation angle $\eta$、maxium DFO $f_d$。

​	为了减小信道开销和信道估计性能，本文提出了**部分估计**信道特征的方法。
$$
\begin{aligned}
\boldsymbol{H}_{i} \mathbf{v} &=\sum_{n=1}^{\bar{N}} \sum_{m=1}^{\bar{M}} \tilde{x}_{n, m} \tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{n}\right) \boldsymbol{a}_{T}^{H}\left(\tilde{\theta}_{T, m}+\beta_{T, m}\right) \mathbf{v} \\
&=\sum_{n=1}^{\bar{N}} x_{n} \tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{n}\right)=\boldsymbol{A}_{R, i}(\boldsymbol{\varphi}) \boldsymbol{x}
\end{aligned}
$$
最终估计的是部分信道信息：${\boldsymbol \varphi}\ \&\ {\boldsymbol x}$

如果每个训练向量都不一样则开销增加$N_q$倍。

在接收端收到的训练向量的信号为：
$$
\boldsymbol{y}=\left[\boldsymbol{H}_{i} \mathbf{v}+\boldsymbol{n}_{i}\right]_{i \in \mathcal{N}_{p}}
$$
被估计的量：

> the estimated partial channel coefficients $\hat{x}$, 
>
> the AoA off-grid vector $\hat{\boldsymbol{\beta}}_{R}$, 
>
> rotation angle $\hat{\eta}$ and 
>
> maximum DFO $\hat{f}_{d}$ 

#### B 用户端的角度域选择性多普勒补偿

利用A中估计得到的$\hat{\boldsymbol{x}}, \hat{\boldsymbol{\beta}}_{R}, \hat{\eta} \text { and } \hat{f}_{d}$ 对高维快速衰落信道进行降维。

1. 在$N$个AoA中选取$N_q$个能量最大的方向，将$\abs{x_n}^2$作为$n-th$ AoA上的信号能量
   - $N_d$是用来在空分复用增益和有效CSI信道开销之间作权衡的变量
   
2. 由于每一个AoA方向均有一个DFO部分：$e^{j 2 \pi f_{d} i \cos \left(\bar{\theta}_{R, n}+\beta_{R, n}+\eta\right)}$， 所以，会有DFO的补偿向量
   - $\tilde{\boldsymbol{a}}_{R, i}^{H}\left(\hat{\boldsymbol{\varphi}}^{n}\right)=\boldsymbol{a}_{R}^{H}\left(\tilde{\theta}_{R, n}+\hat{\beta}_{R, n}\right) \times e^{-j 2 \pi \hat{f}_{d} i \cos \left(\bar{\theta}_{R, n}+\hat{\beta}_{R, n}+\eta\right)}$
   - 选取部分AoA方向：$\mathbf{W}_{i}^{d}=\left[\tilde{\boldsymbol{a}}_{R, i}\left(\hat{\boldsymbol{\varphi}}^{n}\right)\right]_{n \in \mathcal{N}_{d}} \in \mathbb{C}^{N \times N_{d}}$
   - 转换为慢时变信道：$\boldsymbol{H}_{i}^{s}=\left(\mathbf{W}_{i}^{d}\right)^{H} \boldsymbol{H}_{i}$
   
3. ${\boldsymbol H}_i^s$ 是经过多普勒补偿之后的信道信息，

   - 将$\left(\mathbf{W}_{i}^{d}\right)^{H}$ 看作是一个列的向量的向量，将$H_i$ 看作是一个整体，并写为$\sum_{n=1}^{\tilde{N}} \sum_{m=1}^{\tilde{M}} \tilde{x}_{n, m} \tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{n}\right) \boldsymbol{a}_{T}^{H}\left(\tilde{\theta}_{T, m}+\beta_{T, m}\right)$，
   - 可以看作是：

   $$
   \left[\begin{array}{ccccc}
   {\bf w}_{1}  \\
   {\bf w}_{2}  \\
   \vdots  \\
   {\bf w}_{N_d} 
   \end{array}\right]\cdot {\boldsymbol H}_i=
   \left[\begin{array}{ccccc}
   {\bf w}_{1} {\boldsymbol H}_i \\
   {\bf w}_{2} {\boldsymbol H}_i \\
   \vdots  \\
   {\bf w}_{N_d} {\boldsymbol H}_i
   \end{array}\right]
   $$

   - $$
     \begin{array}{r}
     \boldsymbol{H}_{i}^{s}=\sum_{m=1}^{\tilde{M}}\left[\tilde{x}_{n, m}+\sum_{\tilde{n}=1, \tilde{n} \neq n} \tilde{x}_{\tilde{n}, m} \tilde{\boldsymbol{a}}_{R, i}^{H}\left(\boldsymbol{\varphi}^{n}\right) \tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{\tilde{n}}\right)\right]_{n \in \mathcal{N}_{d}} \\
     \times \boldsymbol{a}_{T}^{H}\left(\tilde{\theta}_{T, m}+\beta_{T, m}\right)=\boldsymbol{H}^{s}+\Delta \boldsymbol{H}_{i}
     \end{array}
     $$

   - 由此就将信道信息分为了慢时变信道和快时变信道：

   - 慢时变：$\boldsymbol{H}^{s}=\sum_{m=1}^{\tilde{M}}\left[\tilde{x}_{n, m}\right]_{n \in \mathcal{N}_{d}} \boldsymbol{a}_{T}^{H}\left(\theta_{T, m}+\beta_{T, m}\right)$，可以看到和subframe index $i$ 无关，也就是说该部分在一整个frame中保持恒定。

   - 快时变：$\Delta \boldsymbol{H}_{i}=\sum_{m=1}^{\tilde{M}}\left[\sum_{\tilde{n}=1, \tilde{n} \neq n}^{\tilde{N}} \tilde{x}_{\tilde{n}, m} \tilde{\boldsymbol{a}}_{R, i}^{H}\left(\boldsymbol{\varphi}^{n}\right) \tilde{\boldsymbol{a}}_{R, i}\left(\boldsymbol{\varphi}^{\tilde{n}}\right)\right]_{n \in \mathcal{N}_{d}}$
     $\times \boldsymbol{a}_{T}^{H}\left(\theta_{T, m}+\beta_{T, m}\right)$

4. 快时变组件$\Delta {\boldsymbol H}_i$的二阶矩（方差）大小是遵从$\mathcal{O}\left(\frac{L}{N^{2}}\right)$ 可以看到：

   - 增加接收天线的规模可以使其减小
   - 多径增加会使其增大，接收信号更不稳定。
   - 当$N$足够大，$\Delta {\boldsymbol H}_i$ 的能量可以被忽略

5. 大规模MIMO

   - 角度域方法应用了大规模MIMO中的渐变（asymptotical）特性
   - 更大规模的接收机天线会带来更高的空间分辨率，从而提取不同AoA中的多普勒特性
   - 当$N$足够大，阵列相应中的${\boldsymbol \alpha}_R(\theta)$相互正交， $\Delta {\boldsymbol H}_i$ 的能量可以被忽略

#### C BS端慢时变信道估计

在经典场景下，基站处的射频链路数量是基站天线数量的$1/2\ \text{or}\ 1/4$ 。因此在上行链路估计中，用户可以$2N_d\ \text{or}\ 4N_d$ 个正交的训练向量（pilots）。这些信号会帮助基站估计信道信息：${\boldsymbol H}_i^s$ 。BS基于估计的${\boldsymbol H}_i^s$ 设计precoding策略，这种策略可以用于both上行和下行。

1. ${\boldsymbol H}_i^s$的相关时间比多普勒补偿之后的符号时间（symbol durations）要大得多[^1-22] 所以每一个frame的时长必须要比${\boldsymbol H}_i^s$的相干时间要短。
2. 因此，每个frame可以容纳的symbol数量远远大于$N_d$（有效AoA的数量，代表开销数量级），所以，本文提出的方案是可以在实际系统中所接受的。

#### D BS端的训练向量设计





[^1]: G. Liu, A. Liu, R. Zhang and M. Zhao, "Angular-Domain Selective Channel Tracking and Doppler Compensation for High-Mobility mmWave Massive MIMO," in IEEE Transactions on Wireless Communications, vol. 20, no. 5, pp. 2902-2916, May 2021, doi: 10.1109/TWC.2020.3045272.
[^1-21]: [W. U. Bajwa, J. Haupt, A. M. Sayeed, and R. Nowak, “Compressed channel sensing: A new approach to estimating sparse multipath channels,” Proc. IEEE, vol. 98, no. 6, pp. 1058–1076, Jun. 2010.](doc/1-21_channel.md)
[^1-24]: J. Dai, A. Liu, and V. K. N. Lau, “FDD massive MIMO channel estimation with arbitrary 2D-array geometry,” IEEE Trans. Signal Process., vol. 66, no. 10, pp. 2584–2599, May 2018.

[^1-22]: W. Guo, W. Zhang, P. Mu, F. Gao, and H. Lin, “High-mobility wideband massive MIMO communications: Doppler compensation, analysis and scaling laws,” IEEE Trans. Wireless Commun., vol. 18, no. 6, pp. 3177–3191, Jun. 2019







>  # Noted
>
> 绘图采用liboffice绘制

