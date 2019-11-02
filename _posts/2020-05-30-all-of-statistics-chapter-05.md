---
layout: post
title: All of Statistics - Chapter 5 Solutions
date: 2020-05-30 12:00:00-0800
tags: all-of-statistics
---

**Acknowledgements**: Thanks to Ben S. for correcting some mistakes.

## 1.

### a)

See Question 8 of Chapter 3.

### b)

First, note that

$$
\begin{multline}
S_{n}^{2}=\frac{1}{n-1}\sum_{i}\left(X_{i}-\bar{X}\right)^{2}=\frac{1}{n-1}\sum_{i}\left(X_{i}^{2}-2X_{i}\bar{X}+\bar{X}^{2}\right)\\
=\frac{1}{n-1}\sum_{i}X_{i}^{2}-\frac{n}{n-1}\bar{X}^{2}=c_{n}\frac{1}{n}\sum_{i}X_{i}^{2}-d_{n}\bar{X}^{2}
\end{multline}
$$

where $$c_{n}\rightarrow1$$ and $$d_{n}\rightarrow1$$.
By the WLLN, $$n^{-1}\sum_{i}X_{i}^{2}$$ and $$\bar{X}^{2}$$ converge, in probability, to $$\mathbb{E}[X_{1}^{2}]$$ and $$\mu^{2}$$.
By Theorem 5.5 (d), $$c_{n}n^{-1} \sum_{i}X_{i}^{2}$$ and $$d_{n}\bar{X}^{2}$$ converge, in probability, to the same quantities.
Lastly, by Theorem 5.5 (a), $$S_{n}^{2}$$ converges, in probability, to $$\mathbb{E}[X_{1}^{2}]-\mu^{2}=\sigma^{2}$$.

## 2.

Suppose $$X_{n}$$ converges to $$b$$ in quadratic mean.
By Jensen's inequality, 

$$
\mathbb{E}\left[X_{n}-b\right]^{2}
\leq\mathbb{E}\left[\left|X_{n}-b\right|\right]^{2}
\leq\mathbb{E}[(X_{n}-b)^2]
\rightarrow0
$$

Therefore, $$\mathbb{E}X_{n}\rightarrow b$$.
Next, note that

$$
\mathbb{E}[(X_{n}-b)^2]
=\mathbb{E}\left[X_{n}^{2}\right]-2b\mathbb{E}\left[X_{n}\right]+b^{2}
=\mathbb{V}(X_{n})+\mathbb{E}[X_{n}]^{2}-2b\mathbb{E}\left[X_{n}\right]+b^{2}
$$

Taking limits of both sides reveals $$\lim_{n}\mathbb{V}(X_{n})=0$$.
As for the converse, we can apply the limits $$\lim_{n}\mathbb{E}[X_{n}]=b$$ and $$\lim_{n}\mathbb{V}(X_{n})=0$$ directly to the equation above.

## 3.

Since the expectation of $$\bar{X}$$ is $$\mu$$ and the variance of $$\bar{X}$$ converges to zero, the desired result is obtained by an application of our findings in Problem 2.

Alternatively, taking a more direct approach, note that

$$
\begin{multline}
\mathbb{E}\left[\left(\bar{X}-\mu\right)^{2}\right]=\mathbb{E}\left[\bar{X}^{2}-2\mu\bar{X}+\mu^{2}\right]=\mathbb{E}\left[\bar{X}^{2}\right]-\mu^{2}\\
=\frac{1}{n^{2}}\mathbb{E}\left[\sum_{i}X_{i}^{2}+\sum_{i,j\colon i\neq j}X_{i}X_{j}\right]-\mu^{2}=\frac{1}{n}\mathbb{E}\left[X_{1}^{2}\right]+\frac{n-1}{n}\mathbb{E}\left[X_{1}X_{2}\right]-\mu^{2}.
\end{multline}
$$

Taking the limit,

$$
\mathbb{E}[(\bar{X}-\mu)^{2}]
\rightarrow\mathbb{E}\left[X_{1}X_{2}\right]-\mu^{2}
=\mathbb{E}\left[X_{1}\right]\mathbb{E}\left[X_{2}\right]-\mu^{2}
=0.
$$

## 4.

Let $$\epsilon>0$$.
For $$n$$ sufficiently large,

$$
\mathbb{P}(\left|X_{n}-0\right|>\epsilon)=\mathbb{P}(X_{n}>\epsilon)=\mathbb{P}(X_{n}=n)=1/n^{2}\rightarrow0
$$

and hence $$X_{n}$$ converges in probability.
However,

$$
\mathbb{E}\left[\left(X_{n}-0\right)^{2}\right]=\mathbb{E}\left[X_{n}^{2}\right]\ge\mathbb{E}\left[X_{n}^{2}I_{\{X_{n}=n\}}\right]=n^{2}\mathbb{P}(X_{n}=n)=1
$$

and hence $$X_{n}$$ does not converge in quadratic mean.

## 5.

It is sufficient to prove the second claim since convergence in quadratic mean implies convergence in probability.
Similarly to Problem 3, we can define $$Y_i = X_i^2$$ and apply our findings in Problem 2 to $$\bar{Y}$$.

Alternatively, taking a more direct approach, note that

$$
\left(\frac{1}{n}\sum_{i}X_{i}^{2}-p\right)^{2}=\frac{1}{n^{2}}\sum_{i}X_{i}^{4}+\frac{1}{n^{2}}\sum_{i,j\colon i\neq j}X_{i}^{2}X_{j}^{2}-\frac{2}{n}p\sum_{i}X_{i}^{2}+p^{2}.
$$

Taking expectations, and using the fact that $$X_i^k = X_i$$ and $$\mathbb{E} X_i = p$$,

$$
\begin{multline}
\mathbb{E}\left[\left(\frac{1}{n}\sum_{i}X_{i}^{2}-p\right)^{2}\right]=\frac{1}{n}\mathbb{E}\left[X_{1}^{4}\right]+\frac{n-1}{n}\mathbb{E}\left[X_{1}^{2}\right]\mathbb{E}\left[X_{2}^{2}\right]-2p\mathbb{E}\left[X_{1}^{2}\right]+p^{2}\\
=\frac{1}{n}p+\frac{n-1}{n}p^{2}-p^{2}\rightarrow p^{2}-p^{2}=0.
\end{multline}
$$

## 6.

Letting $$F$$ denote the CDF of a standard normal distribution, by the CLT,

$$
\begin{multline}
\mathbb{P}\left(\frac{X_{1}+\cdots+X_{100}}{100}\geq68\right)\\
=\mathbb{P}\left(\frac{\sqrt{100}}{2.6}\left(\frac{X_{1}+\cdots+X_{100}}{100}-68\right)\geq0\right)\approx 1-F(0)=0.5.
\end{multline}
$$

## 7.

Let $$f>0$$ be a function and $$\epsilon>0$$ be a constant.
Then,

$$
\mathbb{P}(\left|f(n)X_{n}-0\right|>\epsilon)=\mathbb{P}(X_{n}>\epsilon/f(n))\leq\mathbb{P}(X_{n}\neq0)=1-\exp(-1/n)\rightarrow0.
$$

It follows that $$f(n) X_n$$ converges to zero in probability.
Take $$f = 1$$ for Part (a) and $$f(n) = n$$ for (b).

## 8.

Letting $$F$$ denote the CDF of a standard normal distribution, by the CLT,

$$
\begin{multline}
\mathbb{P}(Y<90)=\mathbb{P}(X_{1}+\cdots+X_{100}<90)\\
=\mathbb{P}\left(\frac{\sqrt{100}}{1}\left(\frac{X_{1}+\cdots+X_{100}}{100}-1\right)<-1\right)\approx F(-1)
\end{multline}
$$

## 9.

Let $$\epsilon>0$$.
Then,

$$
\mathbb{P}(\left|X_{n}-X\right|>\epsilon)\leq\mathbb{P}(X_{n}\neq X)=1/n\rightarrow0.
$$

Therefore, $$X_{n}$$ converges in probability (and hence in distribution) to $$X$$.
On the other hand,

$$
\begin{multline}
\mathbb{E}\left[\left(X-X_{n}\right)^{2}\right]=\mathbb{E}\left[\left(X-e^{n}\right)^{2}I_{\{X_{n}\neq X\}}\right]\\
=\mathbb{E}\left[1-2Xe^{n}+e^{2n}\right]\mathbb{P}(X_{n}\neq X)=\frac{1+e^{2n}}{n}\rightarrow\infty.
\end{multline}
$$

## 10.

Since $$1\leq x^{k}/t^{k}$$ whenever $$x\geq t>0$$, it follows that

$$
\mathbb{P}(Z>t)
=\mathbb{E}\left[I_{\{Z > t\}}\right]
\leq\mathbb{E}\left[I_{\{Z > t\}} \left(\frac{Z}{t}\right)^k \right]
\leq\frac{\mathbb{E}\left[I_{\{Z > t\}} \left|Z\right|^k \right]}{t^k}
$$

Therefore, since the distribution is symmetric,

$$
\mathbb{P}(|Z|>t)
=2\mathbb{P}(Z>t)
\leq\frac{\mathbb{E}\left[\left|Z\right|^{k}\left(I_{\{Z>t\}}+I_{\{Z<-t\}}\right)\right]}{t^{k}}
\leq\frac{\mathbb{E}\left|Z\right|^{k}}{t^{k}}.
$$

Note that we only used symmetry in establishing the above and hence the result is more general than the problem description implies.

## 11.

First, note that $$X$$ is almost surely zero.
Let $$\epsilon>0$$ and $$Z$$ be a standard normal random variable.
Then,

$$
\mathbb{P}(\left|X_{n}-X\right|>\epsilon)=\mathbb{P}(\left|X_{n}\right|>\epsilon)=\mathbb{P}(\left|Z\right|>\epsilon\sqrt{n})\leq\frac{\mathbb{E}\left[Z^{2}\right]}{\epsilon^{2}n}=\frac{1}{\epsilon^{2}n}\rightarrow0.
$$

Therefore, $$X_{n}$$ converges in probability (and hence in distribution) to zero.

## 12.

Let $$F$$ be the CDF of an integer valued random variable $$K$$.
Let $$k$$ be an integer.
It follows that $$F(k)=F(k+c)$$ for all $$0\leq c<1$$.
We use this observation multiple times below.

To prove the forward direction, suppose $$X_{n}\rightsquigarrow X$$.
By definition, $$F_{X_{n}}\rightarrow F_{X}$$ at all points of continuity of $$F_{X}$$.
Therefore,

$$
\begin{multline}
\mathbb{P}(X_{n}=k)=F_{X_{n}}(k+1/2)-F_{X_{n}}(k-1/2)\rightarrow F_{X}(k+1/2)-F_{X}(k-1/2)\\=\mathbb{P}(X=k).
\end{multline}
$$

To prove the reverse direction, suppose $$\mathbb{P}(X_{n}=k)\rightarrow\mathbb{P}(X=k)$$ for all integers $$k$$.
Let $$j$$ be an integer and note that

$$
F_{X_{n}}(j)=\sum_{k\leq j}\mathbb{P}(X_{n}=k)\rightarrow\sum_{k\leq j}\mathbb{P}(X=k)=F_{X}(j)
$$

and hence $$X_{n}\rightsquigarrow X$$ as desired.

## 13.

First, note that

$$
F_{X_{n}}(x)=\mathbb{P}(\min\left\{ Z_{1},\ldots,Z_{n}\right\} \leq x/n)=1-\mathbb{P}(Z_{1}\geq x/n)^{n}.
$$

If $$x\leq0$$, it follows that $$F_{X_{n}}(x)=0$$. Otherwise,

$$
\begin{multline}
\mathbb{P}(Z_{1}\geq x/n)^{n}=\left(1-\mathbb{P}(Z_{1}\leq x/n)\right)^{n}=\left(1-\int_{0}^{x/n}f(z)dz\right)^{n}\\
=\left(1-f(c_{n})\frac{x}{n}\right)^{n}=\left(e^{-f(c_{n})x/n}+O(n^{-2})\right)^{n}\rightarrow e^{-\lambda x}.
\end{multline}
$$

Therefore, $$F_{X_{n}}(x)\rightarrow(1-e^{-\lambda x})I_{(0,\infty)}(x)$$ and hence $$X_{n}$$ converges in distribution to an $$\operatorname{Exp}(\lambda)$$ random variable.
## 14.

By the CLT

$$
\frac{\sqrt{n}}{\sigma}\left(\bar{X}-\mu\right)=\frac{\sqrt{n}}{1/\sqrt{12}}\left(\bar{X}-\frac{1}{2}\right)\rightsquigarrow N(0,1).
$$

Let $$g(x)=x^{2}$$ so that $$g^{\prime}(x)=2x$$. By the delta method,

$$
\frac{\sqrt{n}}{\left|g^{\prime}(\mu)\right|\sigma}\left(g(\bar{X})-g(\mu)\right)=\frac{\sqrt{n}}{1/\sqrt{12}}\left(Y_{n}-\frac{1}{4}\right)\rightsquigarrow N(0,1).
$$

## 15.

Define $$g:\mathbb{R}^{2}\rightarrow\mathbb{R}$$ by $$g(x)=x_{1}/x_{2}$$.
Then, $$\nabla g(x)=(1/x_{2},-x_{1}/x_{2}^{2})^{\intercal}$$.
Define $$\nabla_{\mu}=\nabla g(\mu)$$ for brevity.
By the multivariate delta method,

$$
\sqrt{n}\left(Y_{n}-\frac{\mu_{1}}{\mu_{2}}\right)\rightsquigarrow N(0,\nabla_{\mu}^{\intercal}\Sigma\nabla_{\mu})=N(0,\Sigma_{11}/\mu_{2}^{2}-2\Sigma_{12}\mu_{1}/\mu_{2}^{3}+\Sigma_{22}\mu_{1}^{2}/\mu_{2}^{4}).
$$

## 16.

Let $$X_{n},X,Y\sim N(0,1)$$ be IID with $$X_{n}=Y_{n}$$.
Trivially, $$X_{n}\rightsquigarrow X$$ and $$Y_{n}\rightsquigarrow Y$$.
However, $$\mathbb{V}(X_{n}+Y_{n})=\mathbb{V}(2X_{n})=4$$ while $$\mathbb{V}(X+Y)=2$$ and hence $$X_{n}+Y_{n}$$ does not converge in distribution to $$X+Y$$.
