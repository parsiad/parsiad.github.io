---
layout: post
title: All of Statistics - Chapter 6 Solutions
date: 2020-01-20 12:00:00-0800
tags: all-of-statistics
---

## 1.

Since $$\mathbb{E}_\lambda[\hat{\lambda}] = \mathbb{E}_\lambda[X_1]$$, the estimator is unbiased.
Moreover, $$\operatorname{se}(\hat{\lambda})^2 = \mathbb{V}_\lambda(X_1) / n = \lambda / n$$.
By the bias-variance decomposition, the MSE is equal to $$\operatorname{se}(\hat{\lambda})^2$$.

## 2.

If $$y$$ is between $$0$$ and $$\theta$$,

$$
\begin{equation}
	\mathbb{P}_\theta(\hat{\theta} \leq y)
	= \mathbb{P}_\theta(X_1 \leq y)^n
	= (y/\theta)^n.
\end{equation}
$$

Differentiating yields the PDF of $$\hat{\theta}$$ between $$0$$ and $$\theta$$ as $$y \mapsto n(y/\theta)^n / y$$.
Therefore,

$$
\begin{equation}
	\mathbb{E}_\theta[\hat{\theta}]
	= \int_0^\theta n(y/\theta)^n dy
	= \theta n / (n + 1).
\end{equation}
$$

It follows that the bias of this estimator is $$-\theta/(n+1)$$ Moreover,

$$
\begin{equation}
	\operatorname{se}(\hat{\theta})^2
	= \int_0^\theta ny(y/\theta)^n dy - \mathbb{E}_\theta[\hat{\theta}]^2
	= \theta^2 n / (n+2) - \mathbb{E}_\theta[\hat{\theta}]^2.
\end{equation}
$$

By the bias-variance decomposition, the MSE is $$\theta^2 n / (n+2) - \theta^2 (n^2 - 1) / (n+1)^2$$.

*Remark*. $$\hat{\theta} (n+1)/n$$ is an unbiased estimator.

## 3.

Since $$\mathbb{E}_\theta[\hat{\theta}] = 2 \mathbb{E}_\theta[X_1] = \theta$$, the estimator is unbiased.
Moreover,

$$
\begin{equation}
	\operatorname{se}(\hat{\theta})^2
	= 4 \mathbb{V}_\theta(X_1) / n = \theta^2 / (3n).
\end{equation}
$$

By the bias-variance decomposition, the MSE is equal to $$\operatorname{se}(\hat{\theta})^2$$.

