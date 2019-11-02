---
layout: post
title: All of Statistics - Chapter 8 Solutions
date: 2020-01-23 12:00:00-0800
---

## 1.

TODO (Computer Experiment)

## 2.

TODO (Computer Experiment)

## 3.

TODO (Computer Experiment)

## 4.

This is a [stars and bars](https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)) problem (or, equivalently, an "indistinguishable balls in distinct buckets" problem). For example, the configuration `★|★★★||★` corresponds to sampling $$X_1$$ once, sampling $$X_2$$ three times, sampling $$X_3$$ zero times, and sampling $$X_4$$ once. In general, there are $$n$$ stars and $$n-1$$ bars, and hence the total number of configurations is $$(2n - 1)!/(n!(n-1)!)$$.

## 5.

First, note that

$$
\begin{equation}
\mathbb{E}\left[\overline{X}_{n}^{*}\mid X_{1},\ldots,X_{n}\right] =\mathbb{E}\left[X_{1}^{*}\mid X_{1},\ldots,X_{n}\right]=\overline{X}_{n}.
\end{equation}
$$

Therefore, by the tower property, $$\mathbb{E}[\overline{X}_{n}^{*}]=\mathbb{E}[X_{1}]$$. Next, note that

$$
\begin{equation}
\mathbb{V}(\overline{X}_{n}^{*}\mid X_{1},\ldots,X_{n})=\frac{1}{n}\mathbb{V}(X_{1}^{*}\mid X_{1},\ldots,X_{n})=\frac{1}{n^{2}}\sum_{i}\left(X_{i}-\overline{X}_{n}\right)^{2}.
\end{equation}
$$

The above can also be expressed as $$S_{n}(n-1)/n^{2}$$ where $$S_{n}$$ is the unbiased sample variance of $$(X_{1},\ldots,X_{n})$$. Next, note that

$$
\begin{equation}
\mathbb{E}\left[\left(\overline{X}_{n}\right)^{2}\right]=\frac{1}{n^{2}}\mathbb{E}\left[\sum_{i}X_{i}^{2}+\sum_{i\neq j}X_{i}X_{j}\right]=\frac{1}{n}\left(\sigma^{2}+\mu^{2}\right)+\frac{n-1}{n}\mu^{2}=\frac{\sigma^{2}}{n}+\mu^{2}
\end{equation}
$$

where $$\mu = \mathbb{E}[X_{1}]$$ and $$\sigma^{2} = \mathbb{V}(X_{1})$$. Now, recall that for any random variable $$Y$$, 

$$
\begin{equation}
\mathbb{V}(Y\mid\mathcal{H})=\mathbb{E}\left[Y^{2}\mid\mathcal{H}\right]-\mathbb{E}\left[Y\mid\mathcal{H}\right]^{2}.
\end{equation}
$$

Therefore, by the tower property,

$$
\begin{equation}
\mathbb{E}\left[Y^{2}\right]=\mathbb{E}\left[\mathbb{V}(Y\mid\mathcal{H})+\mathbb{E}\left[Y\mid\mathcal{H}\right]^{2}\right].
\end{equation}
$$

Applying this to our setting,

$$
\begin{equation}
\mathbb{E}\left[\left(\overline{X}_{n}^{*}\right)^{2}\right]=\mathbb{E}\left[\frac{n-1}{n^{2}}S_{n}+\left(\overline{X}_{n}\right)^{2}\right]=\frac{2n-1}{n^{2}}\sigma^{2}+\mu^2.
\end{equation}
$$

As such, we can conclude that 

$$
\begin{equation}
  \mathbb{V}(\overline{X}_{n}^{*})
  = \frac{2n-1}{n^{2}} \sigma^{2}
  = \frac{2n-1}{n} \mathbb{V}(\overline{X}_n)
  \sim 2\mathbb{V}(\overline{X}_{n})
\end{equation}
$$

where the asymptotic is in the limit of large $$n$$.

## 6.

TODO (Computer Experiment)

## 7.

### a)

The distribution of $$\hat{\theta}$$ is given in the solution of Question 2 of Chapter 6.

TODO (Computer Experiment)

### b)

Let $$\hat{\theta}^*$$ be a bootstrap resample. Then,

$$
\begin{equation}
  \mathbb{P}(\hat{\theta}^* = \hat{\theta} \mid \hat{\theta})
  = 1 - \mathbb{P}(\hat{\theta}^* \neq \hat{\theta} \mid \hat{\theta})
  = 1 - \left( 1 - 1/n \right)^n
  \rightarrow 1 - \exp(-1) \approx 0.632.
\end{equation}
$$

## 8.

TODO
