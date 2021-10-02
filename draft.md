# Title:



# Abstract:



# Introduction:



# system model:

## A. System Architecture and signal model

​	考虑一个RIS辅助的多用户下行毫米波MIMO通信系统。该系统中BS装备有一个规模为$\sqrt{N} \times \sqrt{N}$的UPA天线阵列。RIS装备有规模为$\sqrt{M} \times \sqrt{M} $的UPA反射阵列。服务用户均为单天线用户。将所考虑的时间段分为不同的传输帧（frame），就像图一所示的那样

![image-20210930165606883](draft.assets/image-20210930165606883.png)

需要注意的是，为了提高信道估计的精度，尤其是第一跳（hoop）的精度，RIS在UPA中心处安放一个单天线的传感器（sensor），该天线具有和RF chain 一样的射频处理能力。

​	在这篇文章中，用户相对于BS和RIS高速移动，会产生Doppler 频偏，因此需要进行频繁的CSI估计；然而考虑到BS和RIS相对静止，因此第一跳（First hoop）的信道相干时间明显大于第二跳（Second hoop），即RIS-user链路的信道相干时间。因此为了减轻用户移动性带来的多普勒效应而产生的性能下降，并降低信道估计开销，我们提出了一种“混合RIS双时间尺度”的信道估计协议。具体来讲，首先将每个Frame分为两个阶段（phase），在第一阶段中只有一个传输帧（subframe），位于RIS的Sensor在帧头处于接收状态，由BS发送导频信号${\bf v} \in \mathbb{C}^{N \times \tau_{BR} }$，从而获得第一跳的CSI，同时根据第一跳已知CSI在每个用户处进行级联信道估计和多普勒补偿；第二阶段中包含$T_{R}$个数据帧（subframe），此时由于BS-RIS link的信道相干时间较长，第一跳的CSI在此后$T_{R}$个数据帧中保持一致，在每一帧帧头只需要重新对级联信道进行估计和多普勒补偿。

​	令$\mathbf{h}_{k} \in \mathbb{C}^{1 \times M}$，$\mathbf{H} \in \mathbb{C}^{M \times N}$， 分别表示RIS到用户和BS到RIS的下行信道，令$\boldsymbol{\Phi}=[\phi_{1}, \cdots, \phi_{m}, \cdots, \phi_{M}]^T \in \mathbb{C}^{M\times 1}$ and $\phi_{m}=e^{j \theta_{m}}$表示RIS反射面的相移矩阵，其幅值始终保持单位一。另外，用${\bf H}_s \in \mathbb{C}^{1\times M}$表示从BS到RIS中心处Sensor的信道矩阵。则RIS处Sensor处接受到的BS导频信号为：
$$
\mathbf{y}_{s}=\mathbf{H}_{s}\mathbf{v}_s+\mathbf{n}_{s}
$$
​	其中，$\mathbf{n}_s$为sensor处接收的AWGN噪声，$\mathbf{n}_{s} \sim \mathcal{C} \mathcal{N}\left(\mathbf{0}, \sigma_{s}^{2} \mathbf{I}_M\right)$，$\sigma_s$为噪声标准差。

​	用户的接收信道可以表示为：
$$
\mathbf{y}_{k}=\mathbf{h}_{k} \operatorname{diag}({\boldsymbol{\Phi})} \mathbf{H}\mathbf{x}+\mathbf{n}_{k}
$$




## B. First hoop channel model



## C. Cascade channel model



FTR



# channel estimation

## A. channel estimation protical





## B. first hoop Channel estimation



## C. Cascade Channel estimation and Doppler compensation for the second hoop





