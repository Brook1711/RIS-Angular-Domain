# variational bayesian 

[Variational Bayesian methods - Wikipedia](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)



**Variational Bayesian methods** are a family of techniques for approximating intractable [integrals](https://en.wikipedia.org/wiki/Integral) arising in [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning). 



In the former purpose (that of approximating a posterior probability), variational Bayes is an alternative to Monte Carlo sampling methods — particularly, Markov chain Monte Carlo methods such as Gibbs sampling — for taking a fully Bayesian approach to statistical inference over complex distributions that are difficult to evaluate directly or sample. In particular, whereas Monte Carlo techniques provide a numerical approximation to the exact posterior using a set of samples, Variational Bayes provides a locally-optimal, exact analytical solution to an approximation of the posterior.

(和蒙特卡龙方法进行对比)



根据 变分：[Variational Bayesian methods - Wikipedia](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
$$
\begin{aligned}
q^*_{\boldsymbol v_i}({\boldsymbol v}_i)&=\frac{\operatorname{exp}(\int_{-{\boldsymbol v}_i}q({\boldsymbol v}) q({\boldsymbol v},{\boldsymbol y}) )  }{\int_{\boldsymbol v_i} \operatorname{exp}\Big(\int_{-{\boldsymbol v}_i}q({\boldsymbol v}) q({\boldsymbol v},{\boldsymbol y}) \Big)  } \\
	&\propto \operatorname{exp}\Bigg(\int_{-{\boldsymbol v}_i}q({\boldsymbol v}) q({\boldsymbol v},{\boldsymbol y}) \Bigg)  
\end{aligned}
$$

$$
q(\boldsymbol v)=
$$


## Update ${\boldsymbol x}_k$



update for $q({\boldsymbol x})\triangleq \prod_{k=1}^K q_k({\boldsymbol x}_k)$
$$
q_{{\boldsymbol x}_k}({\boldsymbol x}_k) = \mathcal{CN}({\boldsymbol x}_k; {\boldsymbol \mu}_k,{\boldsymbol \Sigma}_k)
$$
那么本轮计算过后$q_{\boldsymbol {x}_k}^*({\boldsymbol {x}_k})$中的参数更新为：
$$
{\mathbf \Sigma}_k = \Bigg( \operatorname{diag}\Big( \left\langle \frac{\widetilde{a}_{k,1}}{\widetilde{b}_{k,1}},\dots,\frac{\widetilde{a}_{k,M}}{\widetilde{b}_{k,M}} \right\rangle \Big) +{\mathbf F}_k^H\operatorname{diag}(\boldsymbol{\kappa}) {\mathbf F}_k\Bigg)^{-1}
$$

$$
{\boldsymbol \mu}_k = {\mathbf \Sigma}_k{\mathbf F}_k^H\operatorname{diag}({\boldsymbol \kappa}){\boldsymbol y}_k
$$



开始证明：
$$
q^*_{{\boldsymbol x}_k}({\boldsymbol x}_k)=\int_{-\boldsymbol {x}_k}q({\boldsymbol v}) \operatorname{ln}p({\boldsymbol v},{\boldsymbol y})\ d{\boldsymbol v}
$$




## Update for ${\boldsymbol \gamma}_k$

update for $q({\boldsymbol \gamma})\triangleq \prod_{k=1}^K q({\boldsymbol \gamma}_k)$
$$
q({\boldsymbol \gamma}_k) = \prod_{m=1}^M\Gamma({\gamma}_{k,m};\widetilde{a}_{k,m},\widetilde{b}_{k,m})
$$

$$
\widetilde{a}_{k,m}=\left\langle s_{k,m} \right\rangle a_{k,m}+ \left\langle 1-s_{k,m} \right\rangle 
$$







# Appendex

$$
\boldsymbol{y}_{l, k}=\boldsymbol{\Phi}^{H} \mathbf{V}\left(\omega_{l}\right) \mathbf{D}_{M}\left(\Delta \boldsymbol{\varphi}_{k}\right) \boldsymbol{x}_{k}+\mathbf{N}_{l}, \forall l \in\{1, \ldots, L\}
$$

$\boldsymbol{\xi} \triangleq\left\{\boldsymbol{\xi}_{1}, \boldsymbol{\xi}_{2}, \boldsymbol{\xi}_{3}\right\}$
$$
\begin{aligned}
&\boldsymbol{\xi}_{1}=\left\{\omega_{1}, \ldots, \omega_{L}\right\} \\
&\boldsymbol{\xi}_{2}=\left\{\Delta \varphi_{1}, \ldots, \Delta \varphi_{M}\right\} \\
&\boldsymbol{\xi}_{3}=\left\{\lambda^{c}, p_{01}^{c}, p_{10}^{c}, \mu_{1}^{s}, \sigma_{1}^{s}, \ldots, \mu_{k}^{s}, \sigma_{k}^{s}\right\}
\end{aligned}
$$
包含隐变量的联合概率：
$$
\begin{aligned}
p(\boldsymbol{v}, \boldsymbol{y} ; \boldsymbol{\xi}) &=p(\boldsymbol{y}, \boldsymbol{x}, \boldsymbol{\gamma}, \boldsymbol{s}, \boldsymbol{c}, \boldsymbol{\kappa}) \\
&=p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}) p(\boldsymbol{x} \mid \boldsymbol{\gamma}) p(\boldsymbol{\kappa}) p(\boldsymbol{\gamma} \mid \boldsymbol{s}) p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi}) \\
&=\underbrace{p(\boldsymbol{x} \mid \boldsymbol{\gamma}) p(\boldsymbol{\kappa}) p(\boldsymbol{\gamma} \mid \boldsymbol{s})}_{\text {known distribution }} \underbrace{p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\kappa} ; \boldsymbol{\xi}) p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi})}_{\text {with unknown valuables }}
\end{aligned}
$$


条件概率：
$$
\begin{gathered}
p\left(\boldsymbol{y}_{k} \mid \boldsymbol{x}_{k} ; \boldsymbol{\xi}\right)=C N\left(\boldsymbol{y}_{k} ; \mathbf{F}_{k} \boldsymbol{x}_{k}, \operatorname{Diag}\left(\boldsymbol{\kappa}_{k}\right)^{-1}\right) \\
p(\boldsymbol{y} \mid \boldsymbol{x} ; \boldsymbol{\xi})=\prod^{K} p\left(\boldsymbol{y}_{k} \mid \boldsymbol{x}_{k} ; \boldsymbol{\xi}\right)
\end{gathered}
$$

$$
\begin{aligned}
p(\boldsymbol{c}, \boldsymbol{s} ; \boldsymbol{\xi}) &=p(\boldsymbol{c}) \prod_{k=1}^{\mathrm{n}} p\left(\boldsymbol{s}_{k} \mid \boldsymbol{c}\right) \\
&=p\left(c_{1}\right) \prod_{k=1}^{K} p\left(s_{k, 1} \mid c_{1}\right) \prod_{m=2}^{M}\left[p\left(c_{m} \mid c_{m-1}\right) \prod_{k=1}^{K} p\left(s_{k, m} \mid c_{m}\right)\right]
\end{aligned}
$$

## Factor Graph

[Factor graph - Wikipedia](https://en.wikipedia.org/wiki/Factor_graph)

A factor graph is a [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph) representing the [factorization](https://en.wikipedia.org/wiki/Factorization) of a function. Given a factorization of a function ${\displaystyle g(X_{1},X_{2},\dots ,X_{n})}$
$$
g\left(X_{1}, X_{2}, \ldots, X_{n}\right)=\prod_{j=1}^{m} f_{j}\left(S_{j}\right)
$$
where $S_{j} \subseteq\left\{X_{1}, X_{2}, \ldots, X_{n}\right\}$, the corresponding factor graph $G=(X, F, E)$ consists of variable vertices $X=\left\{X_{1}, X_{2}, \ldots, X_{n}\right\}$, factor vertices $F=\left\{f_{1}, f_{2}, \ldots, f_{m}\right\}$, and edges $E .$ The edges depend on the factorization as follows: there is an undirected edge between factor vertex $f_{j}$ and variable vertex $X_{k}$ if $X_{k} \in S_{j}$. The function is tacitly assumed to be realvalued: $g\left(X_{1}, X_{2}, \ldots, X_{n}\right) \in \mathbb{R}$

Factor graphs can be combined with message passing algorithms to efficiently compute certain characteristics of the function $g\left(X_{1}, X_{2}, \ldots, X_{n}\right)$, such as the marginal distributions.

**message passing algorithms are usually exact for trees, but only approximate for graphs with cycles.**

## Message passing

[Variational message passing - Wikipedia](https://en.wikipedia.org/wiki/Variational_message_passing)

> **Variational message passing** (**VMP**) is an [approximate inference](https://en.wikipedia.org/wiki/Approximate_inference) technique for continuous- or discrete-valued [Bayesian networks](https://en.wikipedia.org/wiki/Bayesian_networks), with [conjugate-exponential](https://en.wikipedia.org/wiki/Conjugate_exponents) parents, developed by John Winn. VMP was developed as a means of generalizing the approximate [variational methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) used by such techniques as [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and works by updating an approximate distribution at each node through messages in the node's [Markov blanket](https://en.wikipedia.org/wiki/Markov_blanket).

定义观测变量${\boldsymbol y }$,隐变量${\boldsymbol v}$，

首先有观测变量本身的似然函数：
$$
\begin{aligned}
\operatorname{ln}p({\boldsymbol y}) &=\int_{\boldsymbol v} q({\boldsymbol v})\operatorname{ln }\Big(\frac{p({\boldsymbol v},{\boldsymbol y})}{p({\boldsymbol v} \mid {\boldsymbol y})}\Big)\\
	&=\int_{\boldsymbol v}q({\boldsymbol v})\bigg[ \operatorname{ln}\frac{p({\boldsymbol v},{\boldsymbol y})}{q({\boldsymbol v})}-\operatorname{ln}\frac{p({\boldsymbol v} \mid {\boldsymbol y})}{q({\boldsymbol v})} \bigg] \\
&= \int_{\boldsymbol v}q({\boldsymbol v})\operatorname{ln}\frac{p({\boldsymbol v},{\boldsymbol y})}{q({\boldsymbol v})}+ \underbrace{(-1) \int_{\boldsymbol v}q({\boldsymbol v})\operatorname{ln}{q({\boldsymbol v})}}_{\text{relative entropy (non-negative)}} \\
	
\end{aligned}
$$
所以有$\operatorname{ln}p({\boldsymbol y})$	的下界(ELBO)：
$$
\int_{\boldsymbol v}q({\boldsymbol v})\operatorname{ln}\frac{p({\boldsymbol v},{\boldsymbol y})}{q({\boldsymbol v})}
$$

当$q({\boldsymbol v})$ 可以被分解为：
$$
q({\boldsymbol v})= \prod_{i} q_i({\boldsymbol v}_i)
$$
where ${\boldsymbol v}_i$ is a disjoint part of the graphical model

This is the key: We can maximize ELOB, or $\mathcal{L}(q)$, by minimizing this special $\mathrm{KL}$ divergence, where we can find approximate and optimal $q_{i}^{*}\left({\boldsymbol v}_{i}\right)$, such that:
$$
\operatorname{ln}q^*_i({\boldsymbol v}_i)=E_{-{\boldsymbol v}_i}[\operatorname{ln}p({\boldsymbol v},{\boldsymbol y})]
$$






### Sum-product rule

[Belief propagation - Wikipedia](https://en.wikipedia.org/wiki/Belief_propagation)

It calculates the [marginal distribution](https://en.wikipedia.org/wiki/Marginal_distribution) for each unobserved node (or variable), conditional on any observed nodes (or variables). 


$$
p(\mathbf{x})=\prod_{a \in F} f_{a}\left(\mathbf{x}_{a}\right)
$$
The algorithm works by passing real valued functions called ***messages*** along with the edges between the hidden nodes. 

More precisely, if *v* is a variable node and *a* is a factor node connected to *v* in the factor graph, the messages from *v* to *a*, (denoted by ${\displaystyle \mu _{v\to a}}$and from *a* to *v* (${\displaystyle \mu _{a\to v}}$), are real-valued functions whose domain is Dom(*v*), the set of values that can be taken by the random variable associated with *v*. These messages contain the "influence" that one variable exerts on another. The messages are computed differently depending on whether the node receiving the message is a variable node or a factor node. Keeping the same notation:


$$
\forall x_{v} \in \operatorname{Dom}(v), \mu_{v \rightarrow a}\left(x_{v}\right)=\prod_{a^{*} \in N(v) \backslash\{a\}} \mu_{a^{*} \rightarrow v}\left(x_{v}\right)
$$

$$
\forall x_{v} \in \operatorname{Dom}(v), \mu_{a \rightarrow v}\left(x_{v}\right)=\sum_{\mathbf{x}_{a}^{\prime}: x_{v}^{\prime}=x_{v}} f_{a}\left(\mathbf{x}_{a}^{\prime}\right) \prod_{v^{*} \in N(a) \backslash\{v\}} \mu_{v^{*} \rightarrow a}\left(x_{v^{*}}^{\prime}\right)
$$
![image-20211121103813880](draft-E-step.assets/image-20211121103813880.png)



upon convergence

1. 每个v节点的发生概率有：

$$
p_{X_{v}}\left(x_{v}\right) \propto \prod_{a \in N(v)} \mu_{a \rightarrow v}\left(x_{v}\right)
$$



2. 每个函数节点的输出概率有：

$$
p_{X_{a}}\left(\mathbf{x}_{a}\right) \propto f_{a}\left(\mathbf{x}_{a}\right) \prod_{v \in N(a)} \mu_{v \rightarrow a}\left(x_{v}\right)
$$

当概率图为tree的时候，可以分部更新，但是当概率图含有loop时，就不一定收敛



## varitional bayes


$$
\operatorname{ln}p({\boldsymbol y})=\underbrace{D_{KL}(q\Vert p)}_{\text{target: minimization}}+\underbrace{\mathcal{L}(q)}_{\text{equivalence: maximization}}
$$
ELBO：
$$
{\mathcal L}(q)= \int_{\boldsymbol v}q({\boldsymbol v})\operatorname{ln}\frac{p({\boldsymbol v},{\boldsymbol y})}{q({\boldsymbol v})}
$$

### Mean field approximation

Factorize $q({\boldsymbol v})$ 
$$
q(\boldsymbol{v})=\prod_{i=1}^{\mathcal Q} q_i({\boldsymbol v}_i\mid {\boldsymbol y})
$$

$$
\operatorname{ln}q_j^*({\boldsymbol v}_j)=\operatorname{E}_{-{\boldsymbol v}_j}\big[ \operatorname{ln}p({\boldsymbol v},{\boldsymbol y}) \big] + \operatorname{constant}
$$

the expectation $\operatorname{E}_{-{\boldsymbol v}_j}\big[ \operatorname{ln}p({\boldsymbol v},{\boldsymbol y}) \big]$  can usually be simplified into a function of the fixed [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter) of the [prior distributions](https://en.wikipedia.org/wiki/Prior_distribution) over the latent variables and of expectations (and sometimes higher [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) such as the [variance](https://en.wikipedia.org/wiki/Variance)) of latent variables not in the current partition (i.e. latent variables not included in
