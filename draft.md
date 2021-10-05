# Title:



# Abstract:



# Introduction:



# system model:

## A. System Architecture and signal model

​	考虑一个RIS辅助的多用户下行毫米波MIMO通信系统。该系统中BS装备有一个规模为$\sqrt{N} \times \sqrt{N}$的UPA天线阵列。RIS装备有规模为$\sqrt{M} \times \sqrt{M} $的UPA反射阵列。服务用户均为装备有$A$个天线的ULA接收端。将所考虑的时间段分为不同的传输帧（frame），就像图一所示的那样

![image-20211002150332940](draft.assets/image-20211002150332940.png)

需要注意的是，为了提高信道估计的精度，尤其是第一跳（hoop）的精度，RIS在UPA中心处安放一个单天线的传感器（sensor），该天线具有和RF chain 一样的射频处理能力。

​	在这篇文章中，用户相对于BS和RIS高速移动，会产生Doppler 频偏，因此需要进行频繁的CSI估计；然而考虑到BS和RIS相对静止，因此第一跳（First hoop）的信道相干时间明显大于第二跳（Second hoop），即RIS-user链路的信道相干时间。为了减轻用户移动性带来的多普勒效应而产生的性能下降，并降低信道估计开销，我们提出了一种“混合RIS双时间尺度”的信道估计协议。具体来讲，首先将每个Frame分为三个阶段（phase），第一阶段和第二阶段被部署在第一个传输帧（subframe）帧头，此时位于RIS的RF chain在帧头处于发射状态，发送导频信号到BS，从而获得第一跳的AoA，该导频信号同时被K个用户接收并估计用户端AoA；紧接着在第二阶段K个用户同时向BS发送正交导频信号，在已知级联信道中部分CSI的情况下进行级联信道估计；在余下的数据帧中由于BS-RIS link的信道相干时间较长，第一跳的CSI在此后$T_{R}$个数据帧中保持一致，在每一帧帧头只需要重新进行第二跳的多普勒补偿即可。

​	令$\mathbf{h}_{k} \in \mathbb{C}^{M \times A}$，$\mathbf{H} \in \mathbb{C}^{N \times M}$， 分别表示用户k到RIS的信道和RIS到BS的信道，令$\boldsymbol{\Phi}=[\phi_{1}, \cdots, \phi_{m}, \cdots, \phi_{M}]^T \in \mathbb{C}^{M\times 1}$ and $\phi_{m}=e^{j \theta_{m}}$表示RIS反射面的相移矩阵，其幅值始终保持单位一。另外，用${\bf H}_{rb} \in \mathbb{C}^{M\times 1}$表示从RIS单天线发射端到BS的信道矩阵。则BS在第一阶段接受到的BS导频信号为：
$$
\mathbf{y}_{rb}=\mathbf{H}_{rb}\mathbf{s}_{r}+\mathbf{n}_{rb}
$$
​	其中，$\mathbf{n}_{rb}$为AWGN噪声，$\mathbf{n}_{rb} \sim \mathcal{C} \mathcal{N}\left(\mathbf{0}, \sigma_{rb}^{2} \mathbf{I}_N\right)$，$\sigma_s$为噪声标准差。$\mathbf{s}_{r}\in \mathbb{C}^{1 \times \tau_{r}}$，为RIS处发射单元在phase 1 中的导频信号其中$s_{r,i} \in \{1,0\}$，且由于系统中只有一个RIS，$\tau_{r}=1$。

​	同时每个用户也接受到了来自RIS的导频信号，令$h_{r,k}\in \mathbb{C}^{1\times A}$表示用户k到RIS处单天线发射端的信道，则第$k$个用户处的接收信号$\mathbf{y}_{r,k}\in \mathbb{C}^{A\times \tau_{r,k}}$为：
$$
\mathbf{y}_{r,k}=\mathbf{h}^{H}_{r,k}\mathbf{s}_{r}+\mathbf{n}_{r,k}
$$
​	其中$\mathbf{n}_{rb}$为AWGN噪声

​	接着，我们假设在phase 2中K个用户同时向BS发射相同长度$\tau_{k,b}$的导频信号$\mathbf{s}_{k,b}\in \mathbb{C}^{A\times \tau_{k,b}}$，为了保证信号的正交性，$\tau_{k,b}\geq K$，则BS接收到的第k个用户的导频信号为$\mathbf{y}_{k,b}\in \mathbb{C}^{N\times\tau_{k,b}}$，
$$
[\mathbf{y}_{k,b}]_{:,t} = \mathbf{H}\operatorname{Diag}({\mathbf\Phi}_t)\mathbf{h}_k\sqrt{p}[\mathbf{s}_{k,b}]_{:,t}+\mathbf{n}_{k,b}
$$
​	级联信道信息可以表示为：${\mathbf G}_k=\{{\mathbf G}_{k,1}, {\mathbf G}_{k,2},\dots,{\mathbf G}_{k,M} \}$
$$
{\mathbf G}_{k,m}=[{\mathbf H}]_{[:,m]}[{\mathbf h}_{k}]_{[m,:]},\quad \forall m\in {1,\dots,M}
$$
从而，接收的导频信道可以进一步写为：
$$
[\mathbf{y}_{k,b}]_{:,t} =\left( \sum_{m=1}^M {\mathbf G}_{k,m}\phi_m \right) \sqrt{p}[\mathbf{s}_{k,b}]_{:,t}+\mathbf{n}_{k,b}
$$




​	在phase 3中，RIS处RF chain发射导频信号$\mathbf{s}_{d}\in\mathbb{C}^{1\times \tau_{r,k}}$，用于进行Doppler补偿。此时用户处的接收信号$\mathbf{y}_{d,k}\in \mathbb{C}^{A\times \tau_{r,k}}$为：
$$
\mathbf{y}_{d,k}=\mathbf{h}^{H}_{r,k}\mathbf{s}_{d}+\mathbf{n}_{r,k}
$$




## B. Channel model

由于BS、RIS、user均装备ULA，则构建空间毫米波信道模型[^2-2][^2-3]：
$$
\mathbf{H}=\sum_{l=1}^{L}\alpha_l\mathbf{a}_N(\psi_l)\mathbf{a}_M^H(\omega_l)\in \mathbb{C}^{N\times M}
$$

$$
\mathbf{h}_k= \sum_{j=1}^{J_k}\beta_{k,j}\mathbf{a}_M(\varphi_{k,j})\mathbf{a}_A^H(\phi_{k,j})\in \mathbb{C}^{M\times A}
$$

$$
\mathbf{H}_{rb} = \sum_{l=1}^{L}\alpha_l\mathbf{a}_N({\psi})\in \mathbb{C}^{N\times 1}
$$

$$
\mathbf{h}_{r,k} = \sum_{j=1}^{J_k}\beta_{k,j}\mathbf{a}_A(\phi_{k,j})\in \mathbb{C}^{A\times 1}
$$



其中，由于信道互易性[^1-5] ，同一链路中相同发射/接收端上行信道和下行信道中具有相同的AoA/AoD，因此定义中不再区分到达和离开，例如，上行信道中BS的AoA等于下行信道中的AoD，于是定义该角度为$\psi_l, \forall l \in \{1,2,\dots, L\}$。$L,J_k$分别表示信道$\mathbf{H},\mathbf{h}_k$中的多径数量。$\alpha_l,\beta_{k,j}$表示对应路径中的pathloss。$\mathbf{H}_{rb}, \mathbf{h}_{r,k}$表示当RIS发射端仅为单天线时的信道模型。需要注意的是，他们与$\mathbf{H},\mathbf{h}_k$共享同样的多径系数：$\psi_l, \phi_{k,j}$和对应的path loss：$\alpha_l,\beta_{k,j}$。这是因为RIS处的单天线发射端放置在RIS的几何中心，在大尺度的远场通信中和ULA的反射元件拥有相同的多径传播路径。利用该单天线的发射特性可以更加精确的估计单hop信道中的信道系数从而达到更精准的级联信道估计。

为简化阵列响应展开式，将ULA阵列响应统一写成以下形式[^2-2]:
$$
\mathbf{a}_{X}(x)=\left[1, e^{-\mathrm{i} 2 \pi x}, \ldots, e^{-\mathrm{i} 2 \pi(X-1) x}\right]^{\mathrm{T}}
$$
where $X \in\{M, N\}$ and $x \in\left\{\omega_{l}, \psi_{l}, \varphi_{k, j}\right\} . \omega_{l}=\frac{d_{\mathrm{RIS}}}{\lambda_{c}} \cos \left(\theta_{l}\right), \psi_{l}=\frac{d_{\mathrm{BS}}}{\lambda_{c}} \cos \left(\phi_{l}\right)$, and $\varphi_{k, j}=$ $\frac{d_{\mathrm{RIS}}}{\lambda_{c}} \cos \left(\vartheta_{k, j}\right)$ are the directional cosine with $\theta_{l}$ and $\phi_{l}$ denoting the $\mathrm{AoD}$ and AoA of the $l$-th spatial path from RIS to BS, respectively, and $\vartheta_{k, j}$ as the AoA of the $j$-th spatial path from user $k$ to RIS. $\lambda_{c}$ is the carrier wavelength. It should be emphasized here that the channel gains $\alpha_{l}$

## C. Angular Domain Channel Expressions

根据[^1-5] 和中文-95，96，103，在MIMO mmwave系统中，

> 说明MIMO信道中存在角度域的稀疏性，相比传统的时域和频域具有更大的优势

以下写出$\mathbf{H},\mathbf{h}_k$以及$\mathbf{H}_{rb}, \mathbf{h}_{r,k}$的角域表达式
$$
{\mathbf H}={\mathbf A}_N{\mathbf {\mathbf A}{\mathbf A}_M^H}\in {\mathbb C}^{N\times M}
$$
其中：
$$
\begin{aligned}
\mathbf{A}_{N} &=\left[\mathbf{a}_{N}\left(\psi_{1}\right), \ldots, \mathbf{a}_{N}\left(\psi_{L}\right)\right] \in \mathbb{C}^{N \times L} \\
{\mathbf A} &=\operatorname{Diag}\left(\alpha_{1}, \alpha_{2}, \ldots, \alpha_{L}\right) \in \mathbb{C}^{L \times L} \\
\mathbf{A}_{M} &=\left[\mathbf{a}_{M}\left(\omega_{1}\right), \ldots, \mathbf{a}_{M}\left(\omega_{L}\right)\right] \in \mathbb{C}^{M \times L}
\end{aligned}
$$
${\mathbf h}_k$：
$$
\mathbf{h}_k = {\mathbf A}_{M,k}{ \mathbf B}_k{\mathbf A}_{A,k}^H\quad \forall k \in {\mathcal K}
$$

$$
\begin{aligned}
\mathbf{A}_{M, k} &=\left[\mathbf{a}_{M}\left(\varphi_{k, 1}\right), \ldots, \mathbf{a}_{M}\left(\varphi_{k, J_{k}}\right)\right] \in \mathbb{C}^{M \times J_{k}} \\
{\mathbf B}_{k} &=\operatorname{Diag}\left(\beta_{k, 1}, \ldots, \beta_{k, J_{k}}\right)\in \mathbb{C}^{J_{k} \times J_k}\\
\mathbf{A}_{A, k} &=\left[\mathbf{a}_{A}\left(\phi_{k, 1}\right), \ldots, \mathbf{a}_{A}\left(\phi_{k, J_{k}}\right)\right] \in \mathbb{C}^{A \times J_{k}} 


\end{aligned}
$$



$\mathbf{H}_{rb}$
$$
{\mathbf H}_{rb} ={\mathbf A}_N{\mathbf A_{rb}}\in \mathbb{C}^{N\times 1}\\
$$
where ${\mathbf A}_{rb}=[\alpha_{1}, \alpha_{2}, \ldots, \alpha_{L}]^T\in \mathbb{C}^{L\times 1}$



$\mathbf{h}_{r,k}$


$$
{\mathbf h }_{r,k}={\mathbf A}_{A,k}{\mathbf B}_{r,k}\in {\mathbb C}^{A\times 1}
$$
Where ${\mathbf B}_{r,k}=[\beta_{k,1},\dots,\beta_{k,J_k}]^T\in {\mathbb C}^{J_k \times 1}$

> 在没有进行角域分解之前，复杂度为：；角域分解之后，复杂度为：。。。

Lamma 1 角域表达可以在DFT变换后显示稀疏性

当$\varphi^{\prime}_{l}\in\{  \}$

Lamma 2 rotation angle 说明.power leak [^1-1] [^2-2]



Lamma 3 级联估计时存在“权重畸变”效应，使得估计的显著角集合存在误差

结论：本文使用的分步估计具有更好的估计精度

# channel estimation

## A. channel estimation protical

Phase 1: broadcast

Phase1.1: broadcast to BS

Phase1.2: broadcast to user k





Phase 2: cascade channel estimation



Phase 3: Doppler compensation











## B. first hop Channel estimation


$$
\mathbf{y}_{rb}=\mathbf{H}_{rb}\mathbf{s}_{r}+\mathbf{n}_{rb}={\mathbf A}_N{\mathbf A}_{rb}{
\mathbf s}_{r} + {\mathbf n}_{rb}\in{\mathbb C}^{N\times 1}
$$

$$
{\mathbf U}_N{\mathbf y}_{rb}={\mathbf U}_N{\mathbf A}_N{\mathbf A}_{rb}{\mathbf s}_r +{\mathbf n}_{rb}

$$

其物理意义为$L$个path中对应的角域能量分布，其中${\mathbf A}_N=[{\mathbf a}_N(\psi_1),\dots,{\mathbf a}_N(\psi_L)]$，则：
$$
{\mathbf A}_N^{D}={\mathbf U}_N{\mathbf A}_N = [{\mathbf U}_N{\mathbf a}_N(\psi_1),\dots,{\mathbf U}_N{\mathbf a}_N(\psi_L)]
$$
根据Lamma 1,如果${\psi_L}$离散，且刚好分布在grid上，则${\mathbf U}_N{\mathbf a}_N(\psi_l)\ \forall l\in \{1,\dots,L\}$中只有一个元素

> 展开$\psi_L$的取值范围

因此， ${\mathbf U}_N{\mathbf A}_N\in {\mathbb C}^{N\times L}$为一个行稀疏、列满秩的矩阵。

但是，由于在实际系统当中，multi-path中的AoA/AoD分布是连续的，当$\psi_l$分布在离散集合之外时，此时的DFT操作会引起能量泄漏现象[^1-5][^2-2][^1-1][^2-3] 此时需要进行rotatino操作，即在DFT操作之前乘以一个旋转向量${\Phi}_N({\Delta \psi}_l)$。
$$
\boldsymbol{\Phi}_{N}\left(\Delta \psi_{l}\right)=\operatorname{Diag}\left\{1, e^{\mathrm{i} \Delta \psi_{l}}, \ldots, e^{\mathrm{i}(N-1) \Delta \psi_{l}}\right\}, \forall l
$$
在进行DFT操作前先对每个阵列响应进行相位对齐(DFT and Rotation)：
$$
{\mathbf A}_N^{DR} = {\mathbf U}_N{\mathbf A}_N^R = [{\mathbf U}_N{\Phi}_N({\Delta \psi}_1){\mathbf a}_N(\psi_1),\dots,{\mathbf U}_N{\Phi}_N({\Delta \psi}_L){\mathbf a}_N(\psi_L)]
$$
我们定义显著角集合$\Omega_N=\{n_l|\forall l\in \{1,\dots,\hat{L}\}\}$ 其中$n_l$表示第$l$个路径对应在BS处的AoA脚标，$\hat{L}$为系统在信道估计阶段取得的显著角个数，这里为了简化模型采用$\hat{L}= L$。

可以从上述过程中看到，虽然可以通过DFT和rotation两步操作将${\mathbf A}_N$分解为行稀疏列满秩矩阵，但是rotation操作需要事先得知所有${\psi}_l\ \forall l \in \{1,\dots ,L\}$的值，$\psi_l$ 的值可以通过$n_l$获知，获取$n_l$的过程被称为“显著角估计”[^1-5]。目前已有在级联信道中显著角估计的方法，但是如Lamma 3所示，对级联信道直接估计显著角有“能量畸变”问题，并且现存方法大多从接收信号的power peak 确定显著角，然而在



## C. Cascade Channel estimation



## D. Doppler compensation for the second hoop



[^1-1]: Angular-domain selective channel tracking and doppler compensation for high-mobility mmWave massive MIMO
[^1-5]: Virtual Angular-Domain Channel Estimation for FDD Based Massive MIMO Systems With
Partial Orthogonal Pilot Design

[^2-3]: Channel Estimation for IRS-Assisted Millimeter-Wave MIMO Systems：Sparsity-Inspired Approaches
[^2-2]: Channel Estimation for RIS-Aided Multiuser Millimeter-Wave Massive MIMO Systems
