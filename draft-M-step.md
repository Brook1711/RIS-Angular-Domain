# update ${\boldsymbol \xi}$

${\boldsymbol \xi}_1 = \left\{ \omega_1,\dots,\omega_L  \right\}$

${\boldsymbol \xi}_2= \left\{ \Delta\varphi_1, \dots,\Delta \varphi_M \right\}$

${\boldsymbol \xi}_3=\left\{ {\kappa_{1,1}},\dots,\kappa_{k,t}, \dots,\kappa_{K,\tau} \right\}$

${\boldsymbol \xi}_4 =\left\{\lambda^c,p^c_{01}, p^c_{10}, \mu^s_1,\sigma^s_1,\dots, \mu^s_k,\sigma^s_k\right\}$

${\boldsymbol v} \triangleq \left\{ {\boldsymbol x} ,  {\boldsymbol \gamma}, {\boldsymbol c}, {\boldsymbol s}\right\}$


$$
\boldsymbol{\xi}_{j}^{(i+1)}=\boldsymbol{\xi}_{j}^{(i)}+\left.\gamma^{(i)} \frac{\partial u\left(\boldsymbol{\xi}_{j}, \boldsymbol{\xi}_{-j}^{(i)} ; \boldsymbol{\xi}_{j}^{(i)}, \boldsymbol{\xi}_{-j}^{(i)}\right)}{\partial \boldsymbol{\xi}_{j}}\right|_{\boldsymbol{\xi}_{j}=\boldsymbol{\xi}_{j}^{(i)}}
$$

$$
u(\boldsymbol{\xi} ; \dot{\boldsymbol{\xi}})=u^{\mathrm{EM}}(\boldsymbol{\xi} ; \dot{\boldsymbol{\xi}})+\sum_{j \in \mathcal{J}_{c}^{1}} \tau_{j}\left\|\boldsymbol{\xi}_{j}-\dot{\boldsymbol{\xi}}_{j}\right\|^{2}
$$

$$
\begin{aligned}
u^{\mathrm{EM}}(\boldsymbol{\xi} ; \dot{\boldsymbol{\xi}})&=\int p(\boldsymbol{v} \mid \boldsymbol{p}, \dot{\boldsymbol{\xi}}) \ln \frac{p(\boldsymbol{v}, \boldsymbol{p}, \boldsymbol{\xi})}{p(\boldsymbol{v} \mid \boldsymbol{p}, \dot{\boldsymbol{\xi}})} d \boldsymbol{v} \\
	&\approx \int q(\boldsymbol{v} ; \dot{\boldsymbol{\xi}}) \ln \frac{p(\boldsymbol{v}, \boldsymbol{p}, \boldsymbol{\xi})}{q(\boldsymbol{v} ; \dot{\boldsymbol{\xi}})} d \boldsymbol{v}
	
\end{aligned}
$$

首先求$\frac{\partial} {\partial {\boldsymbol \xi}_j}{\hat{u}^{EM}}({\boldsymbol \xi}_j,{\boldsymbol \xi}^{(i)}_{-j};({\boldsymbol \xi}^{(i)}_j,{\boldsymbol \xi}^{(i)}_{-j} )$:
$$
\begin{aligned}
\frac{\partial} {\partial {\boldsymbol \xi}_j}{\hat{u}^{EM}}({\boldsymbol \xi}_j,{\boldsymbol \xi}^{(i)}_{-j};{\boldsymbol \xi}^{(i)}_j,{\boldsymbol \xi}^{(i)}_{-j} ) &= \frac{\partial}{\partial {\boldsymbol \xi}_j} \left \{  \left[ \int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)} )\operatorname{ln}p({{\boldsymbol v}, {\boldsymbol y};{\boldsymbol \xi_j,{\boldsymbol \xi}_{-j}^{(i)}}})d{\boldsymbol v} - \int q\left(\boldsymbol{v} ; \boldsymbol{\xi}_{j}^{(i)}, \boldsymbol{\xi}_{-j}^{(i)}\right) \operatorname{ln}q\left(\boldsymbol{v} ; \boldsymbol{\xi}_{j}^{(i)}, \boldsymbol{\xi}_{-j}^{(i)}\right)d{\boldsymbol v} \right] \right\} \\
	&=\frac{\partial }{\partial {\boldsymbol \xi}_j}\left[ \int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)} )\operatorname{ln}p({{\boldsymbol v}, {\boldsymbol y};{\boldsymbol \xi_j,{\boldsymbol \xi}_{-j}^{(i)}}})d{\boldsymbol v} \right] \\
	&=\int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j} \left[ \operatorname{ln}p({\boldsymbol v},{\boldsymbol y};{\boldsymbol \xi}_j,{\boldsymbol \xi}_{-j}^{(i)}) \right] d{\boldsymbol v}

\end{aligned}
$$
注意到$p({\boldsymbol v}, {\boldsymbol y};{\boldsymbol \xi}_j,{\boldsymbol \xi}_{-j}^{(i)})$ 可以被进一步分解：
$$
\begin{aligned}
p({\boldsymbol v}, {\boldsymbol y};{\boldsymbol \xi}_j,{\boldsymbol \xi}_{-j}^{(i)}) &= \underbrace{p(\boldsymbol{x} \mid \boldsymbol{\gamma}) p(\boldsymbol{\kappa}) p(\boldsymbol{\gamma} \mid \boldsymbol{s})}_{\text {known distribution }} \underbrace{p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}) p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi})}_{\text {with unknown valuables }}  \\
\operatorname{ln}p({\boldsymbol v}, {\boldsymbol y};{\boldsymbol \xi}_j,{\boldsymbol \xi}_{-j}^{(i)}) &=\operatorname{ln}p(\boldsymbol{x} \mid \boldsymbol{\gamma})+\operatorname{ln} p(\boldsymbol{\kappa}) +\operatorname{ln}p(\boldsymbol{\gamma} \mid \boldsymbol{s})\\
		&+\operatorname{ln}p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}_{1,2,3})+\operatorname{ln}p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi}_4)
\end{aligned}
$$
将其带入上式可得：
$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\xi}_{j}} \hat{u}^{E M}\left(\boldsymbol{\xi}_{j}, \boldsymbol{\xi}_{-j}^{(i)} ; \boldsymbol{\xi}_{j}^{(i)}, \boldsymbol{\xi}_{-j}^{(i)}\right) &= \int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j} \left[ \operatorname{ln}p({\boldsymbol v},{\boldsymbol y};{\boldsymbol \xi}_j,{\boldsymbol \xi}_{-j}^{(i)}) \right] d{\boldsymbol v}\\
&=\int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j} \left[ \operatorname{ln}p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}_{1,2,3})+\operatorname{ln}p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi}_4) \right] d{\boldsymbol v} \\

&=
\begin{cases}
\int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j}\operatorname{ln}p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}_{1,2,3})d{\boldsymbol v} &, j\in \left\{1,2\right\}\\
\int q({\boldsymbol v};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j} \operatorname{ln}p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi}_3) d{\boldsymbol v} &,j=4
\end{cases}\\
&=
\begin{cases}
\int q({\boldsymbol x};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j}\operatorname{ln}p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}_{1,2,3})d{\boldsymbol x} &, j\in \left\{1,2\right\}\\
\int q({\boldsymbol c},{\boldsymbol s};{\boldsymbol \xi}_j^{(i)},{\boldsymbol \xi}_{-j}^{(i)})\frac{\partial}{\partial {\boldsymbol \xi}_j} \operatorname{ln}p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi}_3) d{\boldsymbol c}d{\boldsymbol s} &,j=4
\end{cases}

\end{aligned}
$$
首先以求解$\omega_l$为例：
$$
\begin{aligned}
\frac{\partial}{\partial \omega_l}\hat{u}^{EM}\left( {\omega}_{l}, \boldsymbol{\xi}_{-\omega_l}^{(i)} ; {\omega}_{l}^{(i)}, \boldsymbol{\xi}_{-{\omega}_{l}}^{(i)} \right) &= \int q({\boldsymbol x{ ;\omega_{l}^{(i)},{\boldsymbol \xi}_{-\omega_l}^{(i)}}})\frac{\partial}{\partial \omega_l} \operatorname{ln}p({\boldsymbol y} \mid {\boldsymbol x},{\boldsymbol \kappa}; {\omega_l }, {\boldsymbol \xi}_{1,-\omega_l}^{(1)})d{\boldsymbol x}

\end{aligned}
$$
其中：
$$
\begin{aligned}
\frac{\partial}{\partial \omega_l} \operatorname{ln}p({\boldsymbol y} \mid {\boldsymbol x},{\boldsymbol \kappa}; {\omega_l }, {\boldsymbol \xi}_{1,-\omega_l}^{(i)}) &= \frac{\partial}{\partial \omega_l}\sum_{k=1}^{K}\sum_{l=1}^{L}\sum_{t=1}^{\tau}\operatorname{ln}\big [\mathcal{ CN}(y_{k,l,t};{\mathbf F}_{k,l,t}{\boldsymbol x}_k, { \kappa_{k,t}^{-1}}) \big] \\

 &= \frac{\partial}{\partial \omega_l}\sum_{k=1}^{K}\sum_{t=1}^{\tau}\operatorname{ln}\big [\mathcal{ CN}(y_{k,l,t};{\mathbf F}_{k,l,t}{\boldsymbol x}_k, { \kappa_{k,t}^{-1}}) \big] \\
 &=\sum_{k=1}^{K}\sum_{t=1}^{\tau} \frac{\partial}{\partial \omega_l} \operatorname{ln}\big [\mathcal{ CN}(y_{k,l,t};{\mathbf F}_{k,l,t}{\boldsymbol x}_k, { \kappa_{k,t}^{-1}}) \big] \\
 &=\sum_{k=1}^{K}\sum_{t=1}^{\tau} \frac{\partial}{\partial ({\mathbf F}_{k,l,t} {\boldsymbol x}_k)} \operatorname{ln}\big [\mathcal{ CN}(y_{k,l,t};{\mathbf F}_{k,l,t}{\boldsymbol x}_k, { \kappa_{k,t}^{-1}}) \big] \frac{\partial}{\partial \omega_l}\left({\mathbf F}_{k,l,t} {\boldsymbol x}_k \right)
 
\end{aligned}
$$
接下来将被求和项目展开，其中
$$
\begin{aligned}
{\mathbf F}_{k,l,t}{\boldsymbol x}_k& =\Big [{\mathbf \Phi}^H {\mathbf V}(\omega_l){\mathbf D}_M(\Delta {\boldsymbol \varphi}) \Big]_{t,:} {\boldsymbol x}_k\\

	&=\Big [{\mathbf \Phi}^H  \Big]_{t,:}{\mathbf V}(\omega_l){\mathbf D}_M(\Delta {\boldsymbol \varphi}){\boldsymbol x}_k\\
	& = \left[
	\begin{matrix}
	e^{j \vartheta_{t,1}} & \cdots & e^{j \vartheta_{t,M}}
	\end{matrix}
	\right]
	
	\left[
	\begin{matrix}
	{\mathbf a}_M(\omega_l)_1 [{\mathbf U}_M]_{1,:} \\ 
	\vdots \\ 
	{\mathbf a}_M(\omega_l)_M [{\mathbf U}_M]_{M,:}
	\end{matrix}
	\right] {\mathbf D}_M(\Delta {\boldsymbol \varphi}){\boldsymbol x}_k \\
	&=\sum_{m=1}^M e^{j \vartheta_{t,m}}{\mathbf{a}_M(\omega_l)_m[{\mathbf U}_M]_{m,:}} {\mathbf D}_M(\Delta {\boldsymbol \varphi}){\boldsymbol x}_k\\
	&=\sum_{m=1}^M e^{j \vartheta_{t,m}}e^{-j2\pi(M-1)\omega_l}[{\mathbf U}_M]_{m,:} {\mathbf D}_M(\Delta {\boldsymbol \varphi}){\boldsymbol x}_k\\
\frac{\partial}{\partial \omega_l} {\mathbf F}_{k,l,t}{\boldsymbol x}_k &=\sum_{m=1}^M e^{j \vartheta_{t,m}}\frac{\partial}{\partial \omega_l}(e^{-j2\pi(M-1)\omega_l})[{\mathbf U}_M]_{m,:} {\mathbf D}_M(\Delta {\boldsymbol \varphi}){\boldsymbol x}_k\\
	&=\sum_{m=1}^M e^{j \vartheta_{t,m}}
	\big(-j2\pi(M-1)\big)e^{-j2\pi(M-1)\omega_l}
	[{\mathbf U}_M]_{m,:} {\mathbf D}_M(\Delta {\boldsymbol \varphi}){\boldsymbol x}_k
\end{aligned}
$$


现在考虑$\frac{\partial}{\partial\left(\mathbf{F}_{k, l, t} \boldsymbol{x}_{k}\right)} \ln \left[\mathcal{C} \mathcal{N}\left(y_{k, l, t} ; \mathbf{F}_{k, l, t} \boldsymbol{x}_{k}, \kappa_{k, t}^{-1}\right)\right]$

化简模型：$\frac{\partial }{\partial \mu} \operatorname{ln}\left[{\mathcal{CN} }\left(y; \mu, \kappa^{-1}\right) \right]$
$$
\begin{aligned}
\frac{\partial }{\partial \mu} \operatorname{ln}\Big[{\mathcal{CN} }\big(y; \mu, \kappa^{-1}\big) \Big]  &= \frac{\partial }{\partial \Big[{\mathcal{CN} }\big(y; \mu, \kappa^{-1}\big) \Big]}\bigg\{\operatorname{ln}\Big[{\mathcal{CN} }\big(y; \mu, \kappa^{-1}\big) \Big] \bigg\} \frac{\partial }{\partial \mu}\Big[ \mathcal{CN}\big( y;\mu, \kappa^{-1} \big) \Big] \\
	&=\frac{1}{{\mathcal{CN} }\big(y; \mu, \kappa^{-1}\big)}
\frac{\partial }{\partial \mu}\Big[ \mathcal{CN}\big( y;\mu, \kappa^{-1} \big) \Big]
\end{aligned}
$$
