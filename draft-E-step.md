# variational bayesian 

[Variational Bayesian methods - Wikipedia](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)



**Variational Bayesian methods** are a family of techniques for approximating intractable [integrals](https://en.wikipedia.org/wiki/Integral) arising in [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning). 



In the former purpose (that of approximating a posterior probability), variational Bayes is an alternative to Monte Carlo sampling methods — particularly, Markov chain Monte Carlo methods such as Gibbs sampling — for taking a fully Bayesian approach to statistical inference over complex distributions that are difficult to evaluate directly or sample. In particular, whereas Monte Carlo techniques provide a numerical approximation to the exact posterior using a set of samples, Variational Bayes provides a locally-optimal, exact analytical solution to an approximation of the posterior.

(和蒙特卡龙方法进行对比)









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





### Sum-product rule



