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
[\mathbf{y}_{k,b}]_{:,t} = \mathbf{H}\operatorname{Diag}(\mathbf{h}_k)\mathbf{\Phi}_t\sqrt{p}[\mathbf{s}_{k,b}]_{:,t}+\mathbf{n}_{k,b}
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
\mathbf{h}_{r,k} = \sum_{j=1}^{J_k}\beta_{k,j}\mathbf{a}_A^{H}(\phi_{k,j})\in \mathbb{C}^{1\times A}
$$



其中，由于信道互易性[^1-5] ，同一链路中相同发射/接收端上行信道和下行信道中具有相同的AoA/AoD，因此定义中不再区分到达和离开，例如，上行信道中BS的AoA等于下行信道中的AoD，于是定义该角度为$\psi_l, \forall l \in \{1,2,\dots, L\}$。$L,J_k$分别表示信道$\mathbf{H},\mathbf{h}_k$中的多径数量。$\alpha_l,\beta_{k,j}$表示对应路径中的pathloss。$\mathbf{H}_{rb}, \mathbf{h}_{r,k}$表示当RIS发射端仅为单天线时的信道模型。需要注意的是，他们与$\mathbf{H},\mathbf{h}_k$共享同样的多径系数：$\psi_l, \phi_{k,j}$和对应的path loss：$\alpha_l,\beta_{k,j}$。这是因为RIS处的单天线发射端放置在RIS的几何中心，在大尺度的远场通信中和ULA的反射元件拥有相同的多径传播路径。利用该单天线的发射特性可以更加精确的估计单hop信道中的信道系数从而达到更精准的级联信道估计。

为简化阵列响应展开式，将ULA阵列响应统一写成以下形式[^2-2]:
$$
\mathbf{a}_{X}(x)=\left[1, e^{-\mathrm{i} 2 \pi x}, \ldots, e^{-\mathrm{i} 2 \pi(X-1) x}\right]^{\mathrm{T}}
$$
where $X \in\{M, N\}$ and $x \in\left\{\omega_{l}, \psi_{l}, \varphi_{k, j}\right\} . \omega_{l}=\frac{d_{\mathrm{RIS}}}{\lambda_{c}} \cos \left(\theta_{l}\right), \psi_{l}=\frac{d_{\mathrm{BS}}}{\lambda_{c}} \cos \left(\phi_{l}\right)$, and $\varphi_{k, j}=$ $\frac{d_{\mathrm{RIS}}}{\lambda_{c}} \cos \left(\vartheta_{k, j}\right)$ are the directional cosine with $\theta_{l}$ and $\phi_{l}$ denoting the $\mathrm{AoD}$ and AoA of the $l$-th spatial path from RIS to BS, respectively, and $\vartheta_{k, j}$ as the AoA of the $j$-th spatial path from user $k$ to RIS. $\lambda_{c}$ is the carrier wavelength. It should be emphasized here that the channel gains $\alpha_{l}$

## C. Angular Domain Channel Expressions

根据[^1-5] ，在MIMO mmwave系统中，

FTR



# channel estimation

## A. channel estimation protical





## B. first hop Channel estimation



## C. Cascade Channel estimation



## D. Doppler compensation for the second hoop





[^1-5]: Virtual Angular-Domain Channel Estimation for FDD Based Massive MIMO Systems With
Partial Orthogonal Pilot Design

[^2-3]: [2-3]Channel Estimation for IRS-Assisted Millimeter-Wave MIMO Systems：Sparsity-Inspired Approaches
[^2-2]: [2-2]Channel Estimation for RIS-Aided Multiuser Millimeter-Wave Massive MIMO Systems
