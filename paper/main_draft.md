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

![Fig. 1](main_draft.assets/image-20211227194955088.png)

Fig. 1

## A. SA-RIS-Assisted Multiuser MIMO system

​    考虑一个RIS辅助的多用户上行毫米波MIMO通信系统，如图一所示。该系统中BS装备有一个规模为$N \times 1$的ULA天线阵列。RIS装备有规模为$M\times 1$的UPA反射阵列。服务用户均为装备有$A$个天线的ULA接收端。

​     目前已经有许多工作研究在RIS上部署RF阵列【】或是直接在RIS的每个反射原件之后连接一个射频处理模块【】以进行BS-RIS link 以及RIS-user link的更好的估计。但是，这和RIS诞生之初减少能耗、降低成本的初衷相违背。因此，在RIS上部署大规模射频链路和信号处理单元在实际中并不可行。同时，许多工作对pure passive RIS辅助的级联信道估计的算法都具有较高的复杂度[^pure-MP][^PanCunhua][^pure-MP-2]。

​    因此，本文提出了一种在不显著提高硬件和能耗成本的前提下，显著降低RIS辅助无线通信系统级联信道估计难度的hardware-boost 框架——SA-RIS

​	SA-RIS在RIS阵列的几何中心处安放一个单天线发射机。该发射机可以向BS和所有用户进行广播。同时由于该天线所处位置为RIS阵列几何中心，由SA-RIS发出的到达收端的信号可以看作是经历了级联信道中相同的传播路径和多径条件。因此可以通过适当的帧结构设计，通过第一阶段广播通过角域信道估计手段[^AngularEstimation]得知部分级联信道中的信息（component），从而降低级联信道估计维度，提高估计精度。

​	显然，和【】中提出的active RIS相比，由于只有一个射频链路，SA-RIS的硬件成本大大降低。同时，和pure passive RIS相比，由于可以在级联信道估计阶段之前就提前预知一部分信道信息，估计算法的复杂度得以降低，同时估计精度得到提升。

​	令$\mathbf{h}_{k} \in \mathbb{C}^{M \times A}$，$\mathbf{H} \in \mathbb{C}^{N \times M}$， 分别表示用户k到RIS的信道和RIS到BS的信道，令$\boldsymbol{\Phi}=[\phi_{1}, \cdots, \phi_{m}, \cdots, \phi_{M}]^T \in \mathbb{C}^{M\times 1}$ and $\phi_{m}=e^{j \theta_{m}}$表示RIS反射面的相移矩阵，其幅值始终保持单位一。另外，用${\bf H}_{rb} \in \mathbb{C}^{M\times 1}$表示从RIS单天线发射端到BS的信道矩阵。则BS接受到的来自RIS的导频信号$\mathbf{y}_{rb}$为：
$$
\mathbf{y}_{rb}=\mathbf{H}_{rb}\mathbf{s}_{r}+\mathbf{n}_{rb}
$$
​	其中，$\mathbf{n}_{rb}$为AWGN噪声，$\mathbf{n}_{rb} \sim \mathcal{C} \mathcal{N}\left(\mathbf{0}, \sigma_{rb}^{2} \mathbf{I}_N\right)$，$\sigma_s$为噪声标准差。$\mathbf{s}_{r}\in \mathbb{C}^{1 \times \tau_{r}}$，为RIS处发射单元在phase 1 中的导频信号其中$s_{r,i} \in \{1,0\}$，且由于系统中只有一个RIS，$\tau_{r}=1$。

​	同时，令$h_{r,k}\in \mathbb{C}^{1\times A}$表示用户k到SA-RIS处单天线发射端的信道，表示，则第$k$个用户处的接收信号$\mathbf{y}_{r,k}\in \mathbb{C}^{A\times \tau_{r,k}}$为：
$$
\mathbf{y}_{r,k}=\mathbf{h}^{H}_{r,k}\mathbf{s}_{r}+\mathbf{n}_{r,k}
$$
​	其中$\mathbf{n}_{rb}$为AWGN噪声

​	接着，我们假设个用户依次向BS发射相同长度$\tau_{k,b}$的导频信号$\mathbf{s}_{k,b}\in \mathbb{C}^{A\times \tau_{k,b}}$，则BS接收到的第k个用户的导频信号为$\mathbf{y}_{k,b}\in \mathbb{C}^{N\times\tau_{k,b}}$，
$$
[\mathbf{y}_{k,b}]_{:,t} = \mathbf{H}\operatorname{Diag}({\mathbf\Phi}_t)\mathbf{h}_k\sqrt{p}[\mathbf{s}_{k,b}]_{:,t}+\mathbf{n}_{k,b}
$$

## B. Channel model

根据3-D SV模型，构建毫米波信道模型[^PanCunhua-2][^PanCunhua]:
$$
\begin{aligned}
\mathbf{H}=\sum_{l=1}^{L}\alpha_l\mathbf{a}_N(\psi^{\prime}_l)\mathbf{a}_M^H(\omega^{\prime}_l)\in \mathbb{C}^{N\times M} \\

\mathbf{h}_k= \sum_{j=1}^{J_k}\beta_{k,j}\mathbf{a}_M(\varphi^{\prime}_{k,j})\mathbf{a}_A^H(\phi^{\prime}_{k,j})\in \mathbb{C}^{M\times A}\\

\mathbf{H}_{rb} = \sum_{l=1}^{L}\alpha_l\mathbf{a}_N({\psi^{\prime}})\in \mathbb{C}^{N\times 1}\\

\mathbf{h}_{r,k} = \sum_{j=1}^{J_k}\beta_{k,j}\mathbf{a}_A(\phi^{\prime}_{k,j})\in \mathbb{C}^{A\times 1}
\end{aligned}
$$
根据信道互异性[^AngularEstimation]，同一链路中相同发射/接收端上行信道和下行信道中具有相同的AoA/AoD，因此定义中不再区分到达和离开，例如，上行信道中BS的AoA等于下行信道中的AoD，于是定义该角度为$\psi^{\prime}_l, \forall l \in \{1,2,\dots, L\}$。$L,J_k$分别表示信道$\mathbf{H},\mathbf{h}_k$中的多径数量。$\alpha_l,\beta_{k,j}$表示对应路径中的pathloss。$\mathbf{H}_{rb}, \mathbf{h}_{r,k}$表示当RIS发射端仅为单天线时的信道模型。需要注意的是，他们与$\mathbf{H},\mathbf{h}_k$共享同样的多径系数：$\psi^{\prime}_l, \phi^{\prime}_{k,j}$和对应的path loss：$\alpha_l,\beta_{k,j}$。这是因为RIS处的单天线发射端放置在RIS的几何中心，在大尺度的远场通信中和ULA的反射元件拥有相同的多径传播路径。利用该单天线的发射特性可以更加精确的估计单hop信道中的信道系数从而达到更精准的级联信道估计。

为简化阵列响应展开式，将ULA阵列响应统一写成以下形式[^PanCunhua]:
$$
\mathbf{a}_{X}(x)=\left[1, e^{-\mathrm{i} 2 \pi x}, \ldots, e^{-\mathrm{i} 2 \pi(X-1) x}\right]^{\mathrm{T}}
$$
where $X \in\{M, N\}$ and $x \in\{\psi_l,\omega_l,\varphi_{k,j},\phi_{j,k} \}$, $\psi_l = \frac{d_\text{RIS}}{\lambda_c}\operatorname{cos}(\psi_l^{\prime})$.and $\varphi_{k, j}=$ $\frac{d_{\mathrm{RIS}}}{\lambda_{c}} \cos \left(\vartheta_{k, j}\right)$ are the directional cosine with $\theta_{l}$ and $\phi_{l}$ denoting the $\mathrm{AoD}$ and AoA of the $l$-th spatial path from RIS to BS, respectively, and $\vartheta_{k, j}$ as the AoA of the $j$-th spatial path from user $k$ to RIS. $\lambda_{c}$ is the carrier wavelength. It should be emphasized here that the channel gains $\alpha_{l}$

## C. Angular Domain Channel Expressions

根据[^AngularEstimation] 和中文-95，96，103，在MIMO mmwave系统中，信道中存在角度域的稀疏性，相比传统的时域和频域具有更大的优势。

以下写出$\mathbf{H},\mathbf{h}_k$以及$\mathbf{H}_{rb}, \mathbf{h}_{r,k}$的角域表达式：
$$
\begin{aligned}
{\mathbf H}&={\mathbf A}_N{\mathbf {\mathbf A}{\mathbf A}_M^H}\in {\mathbb C}^{N\times M}\\

\mathbf{h}_k &= {\mathbf A}_{M,k}{ \mathbf B}_k{\mathbf A}_{A,k}^H\in \mathbb{C}^{1\times A}\quad \forall k \in {\mathcal K}\\

{\mathbf H}_{rb} &={\mathbf A}_N{\mathbf A_{rb}}\in \mathbb{C}^{N\times 1}\\

{\mathbf h }_{r,k} &={\mathbf A}_{A,k}{\mathbf B}_{r,k}\in {\mathbb C}^{A\times 1}
\end{aligned}
$$
其中
$$
\begin{aligned}
\mathbf{A}_{N} &=\left[\mathbf{a}_{N}\left(\psi^{\prime}_{1}\right), \ldots, \mathbf{a}_{N}\left(\psi^{\prime}_{L}\right)\right] \in \mathbb{C}^{N \times L} \\
{\mathbf A} &=\operatorname{Diag}\left(\alpha_{1}, \alpha_{2}, \ldots, \alpha_{L}\right) \in \mathbb{C}^{L \times L} \\
\mathbf{A}_{M} &=\left[\mathbf{a}_{M}\left(\omega^{\prime}_{1}\right), \ldots, \mathbf{a}_{M}\left(\omega^{\prime}_{L}\right)\right] \in \mathbb{C}^{M \times L}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{A}_{M, k} &=\left[\mathbf{a}_{M}\left(\varphi^{\prime}_{k, 1}\right), \ldots, \mathbf{a}_{M}\left(\varphi^{\prime}_{k, J_{k}}\right)\right] \in \mathbb{C}^{M \times J_{k}} \\
{\mathbf B}_{k} &=\operatorname{Diag}\left(\beta_{k, 1}, \ldots, \beta_{k, J_{k}}\right)\in \mathbb{C}^{J_{k} \times J_k}\\
\mathbf{A}_{A, k} &=\left[\mathbf{a}_{A}\left(\phi^{\prime}_{k, 1}\right), \ldots, \mathbf{a}_{A}\left(\phi^{\prime}_{k, J_{k}}\right)\right] \in \mathbb{C}^{A \times J_{k}} 


\end{aligned}
$$

where ${\mathbf A}_{rb}=[\alpha_{1}, \alpha_{2}, \ldots, \alpha_{L}]^T\in \mathbb{C}^{L\times 1}$, ${\mathbf B}_{r,k}=[\beta_{k,1},\dots,\beta_{k,J_k}]^T\in {\mathbb C}^{J_k \times 1}$

在没有进行角域分解之前，复杂度为：；角域分解之后，复杂度为：。。。

#### Remark 1 

角域表达可以在DFT变换后显示稀疏性，离散角度阵列响应DFT的正交性

$$
\frac{d}{\lambda}\leq \frac{1}{2}
$$
且$\frac{d}{\lambda}$ 越大，角域分辨率越高，最终$\frac{d}{\lambda}= \frac{1}{2}$

且
$$
x_i\in [-0.5,0.5)
$$
$x_i\in\{ \psi^{\prime}_l,\omega^{\prime}_l,\varphi^{\prime}_{k,j},\phi^{\prime}_{k,j} \}$

#### Remark 2 

阵列响应角度规整后，能量达到极值

rotation angle 说明.power leak [^AngularDomain] [^PanCunhua]

定义一组特殊的离散AoA/AoD角度系数，以BS上行AoA/下行AoD为例：$\psi^{g}_{l}\in\{ 0-0.5,\dots,\frac{n-1}{N}-0.5, \dots,\frac{N-1}{N}-0.5 \}$ 。当且仅当$\psi_l^{\prime}=\psi_l^{g}$时，
$$
[{\mathbf A}_N^D]_{n,l}=
\begin{cases}
\sqrt{N}\quad &, n = n_l \\
0 &,others
\end{cases}\quad \forall l \ \in \ \{1,\dots,L \}
$$
此时${\mathbf A}_N^D$每一列仅有一个非零元素，并且$n_l,\psi_l^g$之间的映射关系为：
$$
{\psi}_l^g=\frac{n_l-1}{N}-0.5
$$

$$
\psi_l^g = 
\begin{cases}
\frac{n_l-1}{N}\quad &,\frac{n_l-1}{N} < 0.5 \\
\frac{n_l-1}{N}-1 &, \frac{n_l-1}{N}\geq 0.5
\end{cases}
$$

同时由于是一一映射，也可以反过来写为：
$$
n_l = N(\psi^g_l+0.5)+1
$$



$$
n_l=
\begin{cases}
N\psi_l^g+1\quad &,0\leq \psi_l^g <0.5 \\
N\psi_l^g+N+1 &,-0.5\leq \psi_l^g <0

\end{cases}
$$
可以看到每一列的唯一非零元素的行索引互不相同，所以有推论：${\mathbf A}_N^D={\mathbf U}_N{\mathbf A}_N$为一个行稀疏、列满秩的矩阵，并且${\mathbf A}_N^{D}$每一列正交

#### Remark 3 

级联估计时存在“权重畸变”效应，使得估计的显著角集合存在20dB左右的误差

结论：本文使用的分步估计具有更好的估计精度



# Part 3: Channel Estimation

## A. channel estimation Frame Design

![image-20211227202240141](main_draft.assets/image-20211227202240141.png)

Fig. 2

​	BS和RIS一般部署在建筑物表面，收遮挡不明显，且无相对移动，所以信道相关时间较长[^pure-MP][^pure-MP-2]。然而，RIS-User链路受到遮挡和用户移动性影响较大，改变幅度较大，改变速度较快，。针对本文提出的SA-RIS，我们提出了一种混合时间尺度的帧结构设计，以充分利用BS-RIS的慢衰落特性。具体来讲，该帧结构设计分为三个阶段：第一阶段的广播（Broadcast）、第二阶段的级联信道估计和第三阶段的RIS-User信道校准。

（a）Phase 1: Broadcast

​	第一阶段的广播中，RIS向所有用户（包括BS）发射导频信号$\mathbf{s}_{r}\in \mathbb{C}^{1 \times \tau_{r}}$,其中$s_{r,i} \in \{1,0\}$。通过Part 3 B[^AngularEstimation] 中的角域信道估计方法，可以直接得到BS处上行AoA显著角以及对应显著角的Path loss，从而得到$\mathbf{A}_N,\mathbf{A}_{A,k},\forall k \in \mathcal{K}$和$\mathbf{A}, \mathbf{B}_k,\forall k \in \mathcal{K}$，注意到，该阶段只在每个Frame的开头进行估计。

（b）Phase 2：Cascade Channel Estimation

​    由于SA-RIS只具备单天线，所以通过正常手段无法估计出上行信道中RIS的AoA $\boldsymbol{\omega}$和AoD $\boldsymbol{\varphi}$。然而，根据Phase1中所得到的Partial CSI我们可以将问题formulate为一个参数未知的压缩感知问题[^RobustRecovery]。但是，传统压缩感知问题无法。。。【】所以我们提出了双时间尺度的VBI-EM算法。在Phase 2 完成之后，系统就可以获得本时刻的瞬时CSI。

（c）Phase 3：Channel Tracking

​	考虑到在Phase 2 中已经获得了全部的信道信息，并且BS-RIS衰落较慢，则本部分的Channel Tracking只针对RIS-User链路的未知信道信息。由于BS-RIS链路信息已知，本部分算法复杂度将会极低。



## B. Partial Channel estimation

​	该部分考虑Phase1中的部分信道信息估计，利用SA-RIS处部署的单天线设备，可以精准获知$\mathbf{A}_N,\mathbf{A}_{A,k},\forall k \in \mathcal{K}$和$\mathbf{A}, \mathbf{B}_k,\forall k \in \mathcal{K}$ 。

​	RIS处发射导频信号，并被基站接收到，其信号模型可被表示为：
$$
\mathbf{y}_{rb}=\mathbf{H}_{rb}\mathbf{s}_{r}+\mathbf{n}_{rb}={\mathbf A}_N{\mathbf A}_{rb}{
\mathbf s}_{r} + {\mathbf n}_{rb}\in{\mathbb C}^{N\times 1}
$$
​	对其两边同时做DFT变换可以得到：
$$
{\mathbf U}_N{\mathbf y}_{rb}={\mathbf U}_N{\mathbf A}_N{\mathbf A}_{rb}{\mathbf s}_r +{\mathbf U}_N{\mathbf n}_{rb}
$$
​	其物理意义为$L$个path中对应的角域能量分布，其中${\mathbf A}_N=[{\mathbf a}_N(\psi_1),\dots,{\mathbf a}_N(\psi_L)]$，则：
$$
{\mathbf A}_N^{D}={\mathbf U}_N{\mathbf A}_N = [{\mathbf U}_N{\mathbf a}_N(\psi_1),\dots,{\mathbf U}_N{\mathbf a}_N(\psi_L)]
$$
​	我们定义显著角集合$\Omega_N=\{n_l|\forall l\in \{1,\dots,\hat{L}\}\}$ 其中$n_l$表示第$l$个路径对应在BS处的AoA脚标，$\hat{L}$为系统在信道估计阶段取得的显著角个数，这里为了简化模型采用$\hat{L}= L$。可以从上述过程中看到，虽然可以通过DFT和rotation两步操作将${\mathbf A}_N$分解为行稀疏列满秩矩阵。但是rotation操作需要事先得知所有${\psi^{\prime}}_l\ \forall l \in \{1,\dots ,L\}$的值，$\psi^{\prime}_l$ 的值可以通过$n_l$获知，获取$n_l$的过程被称为“显著角估计”[^AngularEstimation]

> 目前已有在级联信道中显著角估计的方法，但是如Lamma 3所示，对级联信道直接估计显著角有“显著角畸变”问题，于是我们在本文中分别估计单跳信道的显著角，这样使得估计精度增加。

​	具体来讲，我们通过寻找power peak[^PanCunhua-2][^PanCunhua-2][^AngularEstimation] 来获知$\Omega_N$:
$$
\Omega_N = \{\Omega\ |\sum_{n_l\in \Omega}|| [{\mathbf U}_N]_{[n_l,:]}{\mathbf y}_{rb} ||^2 \geq \sum_{n_l\in\Omega^{\prime}} || [{\mathbf U}_N]_{[n_l,:]}{\mathbf y}_{rb} ||^2,\forall \Omega^{\prime}\subset{\mathcal N},|\Omega^{\prime}|=|\Omega|=L \}
$$
注意选取$n_l$时的顺序排列：
$$
n_l-n_j
\begin{cases}
<0 \quad , l<j \\
>0 \quad , l>j
\end{cases}
$$
在角域表达中，每个$n_l$都对应一个离散的grid basis角$\psi_l^g$ [^AngularDomain][^CLoudAssisted][^FDD]：
$$
\psi_l^g =f(n_l)
$$
​	但是，由于在实际系统当中，multi-path中的AoA/AoD分布是连续的，当$\psi^{\prime}_l$分布在离散集合之外时，此时的DFT操作会引起能量泄漏现象[^AngularEstimation][^PanCunhua][^AngularDomain][^PanCunhua-2]。所以需要定义rotation angle 以描述真实角度与grid偏离的角度。

​	则给出$\triangle\psi^{\prime}_l$的定义：$\triangle \psi_l = \psi_l^g-\psi_l$，此时，$\psi$未知，但根据Lamma 2可知，$\psi_l^g$对应的DFT中相应频点能量最大，且$\triangle\psi_l$在范围$[-\pi/N,+\pi/N]$内，于是，$\triangle\psi^{\prime}_l$可由以下方法求得：
$$
\triangle \hat{\psi}_{l}=\arg \max _{\triangle \psi \in\left[-\frac{\pi}{N}, \frac{\pi}{N}\right]}\left\|\left[\mathbf{U}_{N}\right]_{n_l, :} \boldsymbol{\Phi}_{N}(\triangle \psi) \mathbf{y}_{rb}\right\|^{2}
$$
​	由此，结合显著角估计，可以得到上行BS处AoA估计$\hat{\psi}_l = \psi_l^g+\Delta \hat{\psi}_l$ 其中，$\mathbf{\Phi}_N \in \mathbb{C}^{N\times N}$ 为rotation矩阵：
$$
\boldsymbol{\Phi}_{N}\left(\triangle \psi\right)=\operatorname{Diag}\left\{1, e^{\mathrm{i} \triangle \psi}, \ldots, e^{\mathrm{i}(N-1) \triangle \psi}\right\}
$$
​	当${\mathbf s}_r=1\times \sqrt{p}\in{\mathbb R}$，即RIS处单天线发送单符号导频信号时，路损系数的估计值$\hat{{\mathbf A}}_{rb}$为：
$$
\begin{aligned}
\hat{\mathbf A}_{rb}&=\frac{1}{N\sqrt{p}}({\hat{\mathbf A}_{N}^{D}})^{H}{\mathbf y}_{rb}^{D}\\
&= \frac{1}{N\sqrt{p}}({{\mathbf A}_{N}^{D}})^{H}{\mathbf A}_N^{D}{\mathbf A}_{rb}{\mathbf s}_{r}\\

&\approx \frac{1}{N\sqrt{p}}N\mathbf{I}_N\mathbf{A}_{rb}\sqrt{p}\\
&={\mathbf A}_{rb}

\end{aligned}
$$
最终，通过在BS端接收的导频信号${\mathbf y}_{rb}$可以估计出基站侧上行AoA（or 下行AoD）$\psi_l$以及多径衰落系数${\mathbf A}_{rb}$ 

同理，用户侧也在接收RIS端发射的导频信号${\mathbf y}_{r,k}$，此时将k-th用户当作是BS则可以估计得到用户侧上行AoD（or 下行AoA）$\phi_{k,j}$以及多径衰落系数${\mathbf B}_{r,k}$

## C. Cascade Channel estimation

在phase 2中通过用户向BS发送的导频信号估计RIS处的上行AoD（or 下行AoA）$\omega_l$和上行AoA（or 下行AoD）$\varphi_{k,j}$，所有用户依次向BS发送上行导频信号，则BS端接收到$k$-th用户发送的$t$-th导频信号的信号表达为：
$$
\left[{\mathbf y}_{k,b} \right]_{:,t} ={\mathbf A}_N{\mathbf A}{\mathbf A}_M^H{\mathbf \Phi}_t{\mathbf A}_{M,k}{\mathbf B}_k{\mathbf A}_{A,k}\sqrt{p}[{\mathbf s_{k,b}}]_{:,t}+{\mathbf n}^b_{k,t}
$$
根据[^AngularEstimation][^AngularDomain]，在设计$k$-th用户的导频信号${\mathbf s}_{k,b}\in {\mathbb C}^{A\times \tau_{k,b}}$时，可以根据Phase 1中获知的用户端上行AoD$\phi_{k,j}$ 将发射能量集中到显著角集合$\Omega_{A,k}$中，使得RIS端接收的导频信号质量更好。

​	通过使用Phase1中获知的Partial CSI，可以将接收信号表达式化简为：
$$
\begin{aligned}
\frac{1}{\sqrt{p} N } \hat{{\mathbf A}}_N^H \left[{\mathbf y}_{k,b} \right]_{:,t} &\approx{\mathbf I}_L {\mathbf A}{\mathbf A}_M^H{\mathbf \Phi}_t{\mathbf A}_{M,k}{\mathbf B}_k{\mathbf A}_{A,k}[{\mathbf{s}_{k,b}}]_{:,t} + \frac{1}{\sqrt{p} N } \hat{{\mathbf A}}_N^H {\mathbf n}^b_{k,t}\\
\frac{1}{\sqrt{p} N }{\mathbf A}^{-1} \hat{{\mathbf A}}_N^H \left[{\mathbf y}_{k,b} \right]_{:,t} &\approx{\mathbf A}_M^H{\mathbf \Phi}_t{\mathbf A}_{M,k}\underbrace{{\mathbf B}_k{\mathbf A}_{A,k}^H[{\mathbf{s}_{k,b}}]_{:,t}}_{\text{part 1}} + \frac{1}{\sqrt{p} N } {\mathbf A}^{-1} \hat{{\mathbf A}}_N^H {\mathbf n}^b_{k,t}
\end{aligned}
$$
​	接下来介绍上行导频信号设计，从上面的公式可以看到，此时信道中的位置量已经只剩下上行信道中的RISAoD($\{ {\omega_1}, \dots,\omega_L\}$)和RIS处的AoA($\{ {\varphi}_1, \dots,\varphi_{J_k} \}$)。需要注意的是，虽然理论上$\{\beta_{k,j},\forall k\in\mathcal{K},\forall j \in \mathcal{J}_k \}$，可以在phase1在用户处得到，但是BS端无法获知，如果使用backhaul链路，则要占用更多的导频开销，所以此时$\{\beta_{k,j},\forall k\in\mathcal{K},\forall j \in \mathcal{J}_k \}$也是未知量。我们的思想是将接收信号表达式化简为只有$\{ {\varphi}_1, \dots,\varphi_{J_k} \}$和$\{ {\omega_1}, \dots,\omega_L\}$以及$\{\beta_{k,j},\forall k\in\mathcal{K},\forall j \in \mathcal{J}_k \}$为变量的形式，即，等式右边除了${\mathbf A}_M^H$和${\mathbf A}_{M,k}$以及$\mathbf{B}_k$之外都是常数矩阵。

​		所以需要设计上式中的$\text{part}\ 1$：
$$
\begin{aligned}
\text{part 1} &= {\mathbf B}_k{\mathbf A}_{A,k}^H[{\mathbf{s}}_{rb}]_{:,t}\\
&=\left(
\begin{matrix}
\beta_{k,1} &  &\\
& \beta_{k,2}\\
&& \ddots \\
&&& \beta_{k,J_k}
\end{matrix}
\right)
\left(
\begin{matrix}
{\mathbf a}_{A}^H(\phi_{k,1})	\\
{\mathbf a}_{A}^H(\phi_{k,2})	\\
\vdots												\\
{\mathbf a}_{A}^H(\phi_{k,J_k})
\end{matrix}
\right)
[{\mathbf s}_{rb}]_{:,t}
\\
&=\left(
\begin{matrix}
\beta_{k,1}{\mathbf a}_{A}^H(\phi_{k,1})[{\mathbf s}_{rb}]_{:,t}	\\
\beta_{k,2}{\mathbf a}_{A}^H(\phi_{k,2})[{\mathbf s}_{rb}]_{:,t}	\\
\vdots												\\
\beta_{k,J_k}{\mathbf a}_{A}^H(\phi_{k,J_k})[{\mathbf s}_{rb}]_{:,t}
\end{matrix}
\right)

\end{aligned}
$$
正如 Lemma中提到的，$\{{\mathbf a}_A(\phi_{k,1}),\dots,{\mathbf a}_A(\phi_{k,J_k}) \}$在A非常的大时候近似正交[^PanCunhua][^AngularDomain]，利用其正交性设计导频信号${\mathbf s}_{rb}$，为简化表示，令${\mathbf s}_t=[{\mathbf s}_{rb}]_{:,t}$，设计目标在于令part 1为常矩阵方便下一步估计，则${\mathbf s}_t$可以按照以下规则设计：
$$
{\mathbf s}_t=\frac{1}{A\sqrt{||\sum_{j=1}^{J_k}{\mathbf a}_A(\phi_{k,j})||^2}}\cdot \sum_{j=1}^{J_k}{\mathbf a}_A(\phi_{k,j}),\quad \forall t \in \{1,\dots,\tau\}
$$
则part 1中的每一项：
$$
\begin{aligned}
\left[\text{part 1}\right]_{j,:}&=\beta_{k,j}{\mathbf a}_A^H(\phi_{k,j})\frac{1}{A\sqrt{||\sum_{j=1}^{J_k}{\mathbf a}_A(\phi_{k,j})||^2}}\cdot \sum_{j=1}^{J_k}{\mathbf a}_A(\phi_{k,j})\\
&\approx \beta_{k,j}\frac{{\mathbf a}_A^H(\phi_{k,j}) {\mathbf a}_A(\phi_{k,j})}{A\sqrt{||\sum_{j=1}^{J_k}{\mathbf a}_A(\phi_{k,j})||^2}}\\
&= \beta_{k,j}\frac{1}{\sqrt{||\sum_{j=1}^{J_k}{\mathbf a}_A(\phi_{k,j})||^2}}
\triangleq\beta_{k,j}\cdot c_s
\end{aligned}
$$

​	因此，通过对导频信号${\mathbf s}_t$的设计，$\text{part 1}$可以被写为：
$$
\begin{aligned}
\text{part 1} &= c_s\cdot 
\left(
\begin{matrix}
\beta_{k,1}\\
\beta_{k,2}\\
\vdots \\
\beta_{k,J_k}
\end{matrix}
\right)_{(J_k \times 1)}
\end{aligned}
$$
​	则，BS端接收的导频信号可以被进一步表示为：
$$
\begin{aligned}

\frac{1}{c_s\sqrt{p} N }{\mathbf A}^{-1} \hat{{\mathbf A}}_N^H \left[{\mathbf y}_{k,b} \right]_{:,t} &\approx{\mathbf A}_M^H{\mathbf A}_{M,k}{\mathbf v}_k + \frac{1}{c_s \sqrt{p} N } {\mathbf A}^{-1} \hat{{\mathbf A}}_N^H {\mathbf n}^b_{k,t}
\end{aligned}
$$
​	其中${\mathbf v}_k=[\beta_{k,1},\beta_{k,2},\dots,\beta_{k,J_k}]^T \in {\mathbb C}^{J_k \times 1}$，$c_s=\frac{1}{\sqrt{||\sum_{j=1}^{J_k}\beta_{k,j}^{-1}{\mathbf a}_A(\phi_{k,j})||^2}}$

​	接下来，为简化表示，使用${\mathbf y}_t\in \mathbb{C}^{L\times 1}$表示$\frac{1}{c_s\sqrt{p} N }{\mathbf A}^{-1} \hat{{\mathbf A}}_N^H \left[{\mathbf y}_{k,b} \right]_{:,t} $。并且将噪声表示为：${\mathbf n}_t = \frac{1}{c_s \sqrt{p} N } {\mathbf A}^{-1} \hat{{\mathbf A}}_N^H {\mathbf n}^b_{k,t}$

​	之后，我们将其写为：
$$
\begin{aligned}
{\mathbf y}_t & = {\mathbf A}_M^H \operatorname{Diag}({\mathbf \Phi}_t){\mathbf A}_{M,k}{\mathbf v}_k+{\mathbf n}_t\\
& = {\mathbf A}_M^H \operatorname{Diag}({\mathbf A}_{M,k}{\mathbf v}_k){\mathbf \Phi}_t+{\mathbf n}_t
\end{aligned}
$$
​	我们通过共轭转置操作考虑${\mathbf y}_t^H \in {\mathbb C}^{1 \times L}$:
$$
{\mathbf y}_t^H = {\mathbf \Phi}_t^H \operatorname{Diag}({\mathbf A}^*_{M,k}{\mathbf v}_k^*){\mathbf A}_M + {\mathbf n}_t^H
$$
​	我们考虑其中$l-th$要素$[{\mathbf y}_t^H]_{:,l} \in {\mathbb C} $:
$$
\begin{aligned}
{[{\mathbf y}_t^H]_{:,l}} & = {\mathbf \Phi}_t^H \operatorname{Diag}({\mathbf A}^*_{M,k}{\mathbf x}_k^*){\mathbf a}_M(\omega_l) + [{\boldsymbol n}_t^H]_{:,l} \\
& = {\mathbf \Phi}_t^H \operatorname{Diag}({\mathbf a}_M(\omega_l)) {\mathbf A}^*_{M,k}{\mathbf v}_k^* + [{\boldsymbol n}_t^H]_{:,l}
\end{aligned}
$$
​	我们接下来考虑的压缩感知问题考虑在时域上的super sampling，所以我们接下来定义一个measurement vector ${\mathbf Y}_l \triangleq \left[\begin{matrix} [{\mathbf y }_1^H]_{:,l}\\ \vdots \\ {\mathbf y }_\tau^H]_{:,l} \end{matrix}\right] \in {\mathbb C}^{\tau \times 1}$ 
$$
\begin{aligned}
{\mathbf Y}_l & = {\mathbf \Phi}^H \operatorname{Diag}({\mathbf a}_M(\omega_l)) {\mathbf A}^*_{M,k}{\mathbf v}_k^* + \left[\begin{matrix} {\mathbf n}_1^H \\ \vdots \\ {\mathbf n}_\tau^H \end{matrix} \right]_{:,l} \\
&={\mathbf \Phi}^H \operatorname{Diag}({\mathbf a}_M(\omega_l)) \mathbf{a}_M({\boldsymbol \varphi}_k) + \mathbf{N}_l
\end{aligned}
$$
​	为了将角域稀疏性提取出来，我们首先采用DFT变换将$\mathbf{a}_M({\boldsymbol \varphi}_k)$变换到角度域，并通过DFT偏转角公式将$\Delta{\boldsymbol \varphi}_k$提取出来：
$$
\begin{aligned}
{\mathbf Y}_l & = {\mathbf \Phi}^H \operatorname{Diag}({\mathbf a}_M(\omega_l)) \frac{{\mathbf U}^H_M {\mathbf U}_M}{M} {\mathbf a}_M  ({\boldsymbol \varphi}) + {\mathbf N}_l\\
									& = {\mathbf \Phi}^H {\mathbf V}({\omega_l}){\mathbf a}_M^{DFT}({\boldsymbol \varphi}) + {\mathbf N}_l\\
									& = {\mathbf \Phi}^H {\mathbf V}({\omega_l}){\mathbf D}_M({\Delta}{\boldsymbol \varphi}) {\mathbf x}_k+ {\mathbf N}_l 

\end{aligned}
$$
其中${\mathbf D}_M(\Delta {\boldsymbol \varphi})= [D_M(\Delta \varphi_1),\dots,D_M(\Delta \varphi_M)] \in {\mathbb C}^{M\times M}$。${\mathbf x}_k\in \mathbb{C}^{M\times 1}$ 为一个$J_k-$稀疏的vector

其每一列$D_M(\Delta {\varphi_m})$中的$m^{\prime}$-th元素的值为：
$$
D_M(\Delta {\varphi_1},m^{\prime})=\begin{cases}
\begin{aligned}
f_M({2\pi}(\frac{m^{\prime}-m} {M}+\Delta\varphi))\ &, \frac{m-1}{M} < 0.5 \\
f_M({2\pi}(\frac{m^{\prime}-m+M} {M}+\Delta\varphi)) &, \frac{m-1}{M} \geq 0.5
\end{aligned}

\end{cases}
$$
其中[^SpectralCS]：
$$
f_M(x) = \frac{1}{\sqrt{M}}e^{jx(M-1)/2} \frac{\operatorname{sin}(Mx/2)}{\operatorname{sin}(x/2)}
$$


### Problem Formulation



### $\omega_l$ Estimation and Calibration

首先利用上一个Frame的$\varphi^{(t-1)}_{k,j} $进行显著角估计，随后在显著角范围内进行联合校准



## D. Channel Tracking



# Part 4: Turbo-EM







[^pure-MP]: H. Liu, X. Yuan and Y. -J. A. Zhang, "Matrix-Calibration-Based Cascaded Channel Estimation for Reconfigurable Intelligent Surface Assisted Multiuser MIMO," in *IEEE Journal on Selected Areas in Communications*, vol. 38, no. 11, pp. 2621-2636, Nov. 2020, doi: 10.1109/JSAC.2020.3007057.
[^pure-MP-2]: H. Liu, X. Yuan and Y. -J. A. Zhang, "Message-Passing Based Channel Estimation for Reconfigurable Intelligent Surface Assisted MIMO," *2020 IEEE International Symposium on Information Theory (ISIT)*, 2020, pp. 2983-2988, doi: 10.1109/ISIT44484.2020.9173987.
[^PanCunhua-2]:  Channel Estimation for IRS-Assisted Millimeter-Wave MIMO Systems：Sparsity-Inspired Approaches
[^PanCunhua]:  Zhou, Gui, et al. "Channel estimation for RIS-aided multiuser millimeter-wave systems." *arXiv preprint arXiv:2106.14792* (2021).
[^FDD]: J. Dai, A. Liu and V. K. N. Lau, "FDD Massive MIMO Channel Estimation With Arbitrary 2D-Array Geometry," in IEEE Transactions on Signal Processing, vol. 66, no. 10, pp. 2584-2599, 15 May15, 2018, doi: 10.1109/TSP.2018.2807390.
[^CloudAssisted]: A. Liu, L. Lian, V. Lau, G. Liu and M. Zhao, "Cloud-Assisted Cooperative Localization for Vehicle Platoons: A Turbo Approach," in IEEE Transactions on Signal Processing, vol. 68, pp. 605-620, 2020, doi: 10.1109/TSP.2020.2964198.
[^AngularDomain]: G. Liu, A. Liu, R. Zhang and M. Zhao, "Angular-Domain Selective Channel Tracking and Doppler Compensation for High-Mobility mmWave Massive MIMO," in IEEE Transactions on Wireless Communications, vol. 20, no. 5, pp. 2902-2916, May 2021, doi: 10.1109/TWC.2020.3045272.
[^FactorGraph]: F. R. Kschischang, B. J. Frey and H. -. Loeliger, "Factor graphs and the sum-product algorithm," in *IEEE Transactions on Information Theory*, vol. 47, no. 2, pp. 498-519, Feb 2001, doi: 10.1109/18.910572.
[^RobustRecovery]: A. Liu, G. Liu, L. Lian, V. K. N. Lau and M. Zhao, "Robust Recovery of Structured Sparse Signals With Uncertain Sensing Matrix: A Turbo-VBI Approach," in IEEE Transactions on Wireless Communications, vol. 19, no. 5, pp. 3185-3198, May 2020, doi: 10.1109/TWC.2020.2971193.
[^DynamicSparsity]: L. Lian, A. Liu and V. K. N. Lau, "Exploiting Dynamic Sparsity for Downlink FDD-Massive MIMO Channel Tracking," in IEEE Transactions on Signal Processing, vol. 67, no. 8, pp. 2007-2021, 15 April15, 2019, doi: 10.1109/TSP.2019.2896179.
[^SpectralCS]: Duarte, Marco F., and Richard G. Baraniuk. "Spectral compressive sensing." *Applied and Computational Harmonic Analysis* 35.1 (2013): 111-129.
[^AngularEstimation]: P. Zhao, K. Ma, Z. Wang and S. Chen, "Virtual Angular-Domain Channel Estimation for FDD Based Massive MIMO Systems With Partial Orthogonal Pilot Design," in IEEE Transactions on Vehicular Technology, vol. 69, no. 5, pp. 5164-5178, May 2020, doi: 10.1109/TVT.2020.2979916.



