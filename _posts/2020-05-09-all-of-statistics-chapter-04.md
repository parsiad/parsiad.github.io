---
layout: post
title: All of Statistics - Chapter 4 Solutions
date: 2020-05-09 12:00:00-0800
tags: all-of-statistics
---

## 1.

Chebyshev's inequality gives $$\mathbb{P}(\left|X-\mu\right|\geq k\sigma)\leq1/k^{2}$$.
An exact calculation yields instead $$e^{-(1+k)}$$.
To see this, note that $$\beta(\mu\pm k\sigma)=1\pm k$$ and $$1-k<0$$ so that

$$
\begin{equation}
\mathbb{P}(\left|X-\mu\right|\leq k\sigma)
=\mathbb{P}(X\leq\mu+k\sigma)
=F(\mu+k\sigma)
=1-e^{-(1+k)}
\end{equation}
$$

## 2.

$$
\begin{equation}
\mathbb{P}(X\geq2\lambda)
=\mathbb{P}(X-\lambda\geq\lambda)
=\mathbb{P}(\left|X-\lambda\right|\geq\lambda)\leq1/\lambda.
\end{equation}
$$

## 3.

First, note that $$\mathbb{V}(\overline{X})=\mathbb{V}(X_{1})/n=p(1-p)/n$$.
Chebyshev's inequality yields

$$
\mathbb{P}(|\overline{X}-p|>\epsilon)
\leq\frac{p\left(1-p\right)}{n\epsilon^{2}}
\leq\frac{\max\left\{ x\left(1-x\right)\colon0\leq x\leq1\right\} }{n\epsilon^{2}}
=\frac{1}{4n\epsilon^{2}}.
$$

Next, note that

$$
\mathbb{P}(|\overline{X}-p|\geq\epsilon)
=\mathbb{P}(\overline{X}-p\geq\epsilon)+\mathbb{P}(\overline{X}-p\leq-\epsilon).
$$

Let $$Y_{i}=(X_{i}-\mathbb{E}X_{1})/n=(X_{i}-p)/n$$ so that $$\overline{X}-p=\sum_{i}Y_{i}$$.
Then, $$\mathbb{E}Y_{i}=0$$ and $$-p/n\leq Y_{i}\leq(1-p)/n$$.
Hoeffding's inequality yields

$$
\begin{multline}
\mathbb{P}(\overline{X}-p\geq\epsilon)
=\mathbb{P}\left(\sum_{i}Y_{i}\geq\epsilon\right)
\leq\exp\left(-t\epsilon\right)\prod_{i}\exp\left(\frac{t^{2}}{8n^{2}}\right)\\
=\exp\left(\frac{t^{2}}{8n}-t\epsilon\right)
\leq\min_{t>0}\exp\left(\frac{t^{2}}{8n}-t\epsilon\right)
=\exp(-2n\epsilon^{2}).
\end{multline}
$$

Similarly, $$\mathbb{P}(\overline{X}-p\leq-\epsilon)=\mathbb{P}(\sum_{i}(-Y_{i})\geq\epsilon)\leq\exp(-2n\epsilon^{2})$$.
It follows that

$$
\mathbb{P}(|\overline{X}-p|\geq\epsilon)
\leq2\exp(-2n\epsilon^{2})
=\frac{1}{1/2+n\epsilon^{2}+n^{2}\epsilon^{4}+O(n^{4})}
$$

is tighter than the Chebyshev bound for sufficiently large $$n$$.

## 4.

### a)

Applying our findings from Question 3,

$$
\mathbb{P}(p\in C_{n})
=1-\mathbb{P}(p\notin C_{n})
\geq1-2\exp(-2n\epsilon_{n}^{2})
=1-2\exp\left(\log\left(\frac{\alpha}{2}\right)\right)
=1-\alpha.
$$

### b)

TODO (Computer Experiment)

### c)

The length of the interval is $$2\epsilon_{n}$$.
This length is at most $$c>0$$ if and only if $$n\geq2\log(2/\alpha)/c^{2}$$.

TODO (Plot)

## 5.

As per the hint,

$$
\begin{multline}
\mathbb{P}(|Z|>t)
=2\mathbb{P}(Z\geq t)
=\sqrt{\frac{2}{\pi}}\int_{t}^{\infty}\exp\left(-\frac{x^{2}}{2}\right)dx\\
\leq\sqrt{\frac{2}{\pi}}\frac{1}{t}\int_{t}^{\infty}x\exp\left(-\frac{x^{2}}{2}\right)dx
=\sqrt{\frac{2}{\pi}}\frac{1}{t}\exp\left(-\frac{t^{2}}{2}\right).
\end{multline}
$$

## 6.

TODO (Plot)

## 7.

A linear combination of IID normal random variables is itself a normal random variable.
Therefore, $$\overline{X}$$ is a random variable with zero mean and variance $$1/n$$.
Letting $$Z\sim N(0,1)$$, Mill's inequality yields

$$
\begin{equation}
\mathbb{P}(|\overline{X}|\geq t)
=\mathbb{P}(|Z|\geq t\sqrt{n})
\leq\sqrt{\frac{2}{\pi}}\frac{1}{t \sqrt{n}}\exp\left(-\frac{t^{2}n}{2}\right).
\end{equation}
$$

The above is tighter than the Chebyshev bound $$1 / (t^2 n)$$ for sufficiently large $$n$$.
