---
layout: post
title: All of Statistics - Chapter 13 Solutions
date: 2021-05-26 12:00:00-0800
tags: all-of-statistics
---

## 1.

It is easier to work in the multivariate setting for this proof.
In light of this, let $$X_{i}$$ be a random $$p$$ dimensional vector.
Define $$X_{-0}$$ as the $$n\times p$$ matrix whose rows are $$X_{i}^{\intercal}$$.
Augment this matrix to obtain $$X=(e\mid X_{-0})$$ where $$e$$ is the vector of ones, corresponding to a design matrix with a bias column.
Let $$Y$$ be the vector whose coordinates are $$Y_{i}$$.

Using the fact that $$\sum_{i}\hat{\epsilon}_{i}^{2}=\Vert Y-X\hat{\beta}\Vert^{2}$$ and matrix calculus, it is straightforward to show that the RSS is minimized when $$\hat{\beta}$$ is chosen to satisfy the linear system

$$
X^{\intercal}X\hat{\beta}=X^{\intercal}Y.
$$

Note that

$$
X^{\intercal}Y=\begin{pmatrix}e^{\intercal}Y\\
X_{-0}^{\intercal}Y
\end{pmatrix}=\begin{pmatrix}n\overline{Y}\\
X_{-0}^{\intercal}Y
\end{pmatrix}
$$

and

$$
X^{\intercal}X=\begin{pmatrix}n & e^{\intercal}X_{-0}\\
X_{-0}^{\intercal}e & X_{-0}^{\intercal}X_{-0}
\end{pmatrix}.
$$

Let $$\hat{\beta}=(\hat{\beta}_{0}\mid\hat{\beta}_{-0})$$ where $$\hat{\beta}_{0}$$ is a scalar.
The first row of the linear system yields

$$
\hat{\beta}_{0}=\overline{Y}-\frac{1}{n}e^{\intercal}X_{-0}\hat{\beta}_{-0}.
$$

Since $$e^{\intercal}X_{-0}=n\overline{X}$$ when $$p=1$$, the above is equivalent to Eq. (13.6).
Substituting the above into the second row of the linear system yields 

$$
\left(X_{-0}^{\intercal}X_{-0}-\frac{1}{n}X_{-0}^{\intercal}ee^{\intercal}X_{-0}\right)\hat{\beta}_{-0}=X_{-0}^{\intercal}Y-X_{-0}^{\intercal}e\overline{Y}.
$$

If $$p=1$$, the above simplifies to

$$
\left(\sum_{i}X_{i}^{2}-n\overline{X}^{2}\right)\hat{\beta}_{1}=\sum_{i}X_{i}Y_{i}-n\overline{X}\overline{Y}
$$

which, with some work, can be shown to be equivalent to Eq. (13.5).

Next, denoting by $$\hat{\epsilon}$$ the vector with coordinates $$\hat{\epsilon}_{i}$$, we have

$$
\hat{\epsilon}=Y-X\hat{\beta}=MY
$$

where $$M=I-X(X^{\intercal}X)^{-1}X^{\intercal}$$.
Denoting by $$\epsilon$$ the vector with coordinates $$\epsilon_{i}$$ and $$\beta$$ the vector of true coefficients,

$$
\hat{\epsilon}=MY=M(X\beta+\epsilon)=M\epsilon.
$$

Using the fact that $$M$$ is both symmetric and idempotent,

$$
\mathrm{RSS}=\sum_{i}\hat{\epsilon}_{i}^{2}=\hat{\epsilon}^{\intercal}\hat{\epsilon}=\epsilon^{\intercal}M^{\intercal}M\epsilon=\epsilon^{\intercal}M\epsilon.
$$

For brevity, we abuse notation by writing $$\mathbb{E} f$$ to mean $$\mathbb{E}[f\mid X]$$.
Then,

$$
\mathbb{E}\left[\mathrm{RSS}\right]=\mathbb{E}\left[\epsilon^{\intercal}M\epsilon\right]=\operatorname{tr}(M\mathbb{E}\left[\epsilon\epsilon^{\intercal}\right]).
$$

Assuming that $$\epsilon_{i}$$ and $$\epsilon_{j}$$ are independent whenever $$i\neq j$$ yields $$\mathbb{E}[\epsilon\epsilon^{\intercal}]=\sigma^{2}I$$ and hence

$$
\mathbb{E}\left[\mathrm{RSS}\right]=\sigma^{2}\operatorname{tr}(M).
$$

Moreover,

$$
\operatorname{tr}(M)=\operatorname{tr}(I_{n\times n})-\operatorname{tr}(X^{\intercal}X(X^{\intercal}X)^{-1})=\operatorname{tr}(I_{n\times n})-\operatorname{tr}(I_{(p+1)\times (p+1)})=n-\left(p+1\right),
$$

establishing that (13.7) is an unbiased estimator of the noise variance.

## 2.

We continue to use the notation established in the answer to the first exercise.
First, note that

$$
\mathbb{E}Y=\mathbb{E}\left[X\beta+\epsilon\right]=X\beta
$$

and

$$
\mathbb{E}\left[YY^{\intercal}\right]=\mathbb{E}\left[\left(X\beta+\epsilon\right)\left(X\beta+\epsilon\right)^{\intercal}\right]=\mathbb{E}\left[X\beta\beta^{\intercal}X^{\intercal}+2X\beta\epsilon^{\intercal}+\epsilon\epsilon^{\intercal}\right]=X\beta\beta^{\intercal}X^{\intercal}+\sigma^{2}I.
$$

Therefore,

$$
\mathbb{E}\hat{\beta}=\left(X^{\intercal}X\right)^{-1}X^{\intercal}\mathbb{E}\left[Y\right]=\beta
$$

and

$$
\begin{multline*}
\mathbb{E}\left[\hat{\beta}\hat{\beta}^{\intercal}\right]=\mathbb{E}\left[\left(X^{\intercal}X\right)^{-1}X^{\intercal}YY^{\intercal}X\left(X^{\intercal}X\right)^{-1}\right]\\
=\left(X^{\intercal}X\right)^{-1}X^{\intercal}\mathbb{E}\left[YY^{\intercal}\right]X\left(X^{\intercal}X\right)^{-1}=\beta\beta^{\intercal}+\sigma^{2}\left(X^{\intercal}X\right)^{-1}.
\end{multline*}
$$

Combining the above yields

$$
\mathbb{V}(\hat{\beta}\hat{\beta}^{\intercal})=\mathbb{E}\left[\hat{\beta}\hat{\beta}^{\intercal}\right]-\mathbb{E}\left[\hat{\beta}\right]\mathbb{E}\left[\hat{\beta}\right]^{\intercal}=\sigma^{2}\left(X^{\intercal}X\right)^{-1}.
$$

In the univariate case, the form

$$
X^{\intercal}X=\begin{pmatrix}n & n\overline{X}\\
n\overline{X} & \sum_{i}X_{i}^{2}
\end{pmatrix}
$$

can be used to derive a closed form expression for the inverse which in turn yields (13.11) as desired.

## 3.

A univariate regression through the origin is a special case of the multivariate regression seen in Exercise 1.
It has least squares coefficient

$$
\frac{\sum_i X_i Y_i}{\sum_i X_i^2}.
$$

This is well-defined whenever at least one of the $$X_i$$ is nonzero.

The standard error of this coefficient is also a special case of the standard error for the multivariate case seen in Exercise 2.
It is

$$
\frac{\sigma^2}{\sum_i X_i^2}.
$$

Since the least squares estimate is an MLE, it is consistent whenever it is well-defined.

## 4.

Using the fact that $$Y_{i}$$ and $$Y_{i}^{*}$$ are IID,

$$
\begin{align*}
\mathbb{E}\left[\hat{R}_{\mathrm{tr}}(S)\right]-R(S) & =\sum_{i}\mathbb{E}\left[\left(\hat{Y}_{i}(S)-Y_{i}\right)^{2}-\left(\hat{Y}_{i}(S)-Y_{i}^{*}\right)^{2}\right]\\
 & =\sum_{i}\mathbb{E}\left[\hat{Y}_{i}(S)^{2}-2\hat{Y}_{i}(S)Y_{i}+Y_{i}^{2}-\hat{Y}_{i}(S)^{2}+2\hat{Y}_{i}(S)Y_{i}^{*}-\left(Y_{i}^{*}\right)^{2}\right]\\
 & =\sum_{i}-2\mathbb{E}\left[\hat{Y}_{i}(S)Y_{i}\right]+\mathbb{E}\left[Y_{i}^{2}\right]+2\mathbb{E}\left[\hat{Y}_{i}(S)Y_{i}^{*}\right]-\mathbb{E}\left[\left(Y_{i}^{*}\right)^{2}\right]\\
 & =-2\sum_{i}\mathbb{E}\left[\hat{Y}_{i}(S)Y_{i}\right]-\mathbb{E}\left[\hat{Y}_{i}(S)\right]\mathbb{E}\left[Y_{i}\right]\\
 & =-2\sum_{i}\operatorname{Cov}(\hat{Y}_{i}(S),Y_{i}).
\end{align*}
$$

## 5.

Let $$\hat{\delta}=\hat{\beta}_{1}-17\hat{\beta}_{0}$$.
By Theorem 13.8,

$$
\mathbb{V}(\hat{\delta})=\mathbb{V}(\hat{\beta}_{1})+17^{2}\mathbb{V}(\hat{\beta}_{0})-17\operatorname{Cov}(\hat{\beta}_{0},\hat{\beta}_{1})=\frac{\sigma^{2}}{ns_{X}^{2}}\left(1+17\overline{X}+\frac{17^{2}}{n}\sum_{i}X_{i}^{2}\right).
$$

Replacing $$\sigma$$ by $$\hat{\sigma}$$ and taking square roots yields $$\hat{\operatorname{se}}(\hat{\delta})$$.
The Wald statistic is $$W=\hat{\delta}/\hat{\operatorname{se}}(\hat{\delta})$$.

## 6.

TODO (Computer experiment).

## 7.

TODO (Computer experiment).

## 8.

Maximizing $$\mathrm{AIC}$$ is equivalent to minimizing $$-2\sigma^{2}\mathrm{AIC}$$.
This is equivalent to minimizing Mallow's $$C_{p}$$ statistic since

$$
\begin{align*}
-2\sigma^{2}\mathrm{AIC} & =-2\sigma^{2}\ell_{S}+2\left|S\right|\sigma^{2}\\
 & =-2\sigma^{2}\left\{ \frac{n}{2}\log(2\pi)-n\log\sigma-\frac{1}{2\sigma^{2}}\sum_{i}\left(\hat{Y}_{i}(S)-Y_{i}\right)^{2}\right\} +2\left|S\right|\sigma^{2}\\
 & =\text{const.}+\sum_{i}\left(\hat{Y}_{i}(S)-Y_{i}\right)^{2}+2\left|S\right|\sigma^{2}\\
 & =\text{const.}+C_{p}+2\left|S\right|\sigma^{2}.
\end{align*}
$$

## 9.

Choosing the model with the highest AIC is equivalent to choosing
the model with the lowest Mallow's $$C_{p}$$ statistic. The two models
have Mallow's statistics $$C_{p}^{0}=\sum_{i}X_{i}^{2}$$ and $$C_{p}^{1}=[\sum_{i}(X_{i}-\hat{\theta})^{2}]+2$$
with $$\hat{\theta}=\overline{X}$$. Note that

$$
C_{p}^{0}-C_{p}^{1}=\sum_{i}X_{i}^{2}-\sum_{i}\left(X_{i}-\hat{\theta}\right)^{2}+2=n\hat{\theta}^{2}-2.
$$

Therefore, $$\mathcal{M}_{0}$$ is picked if and only if $$\hat{\theta}^2 < 2/n$$.

### a)

First, note that $$\hat{\theta} \sim N(\theta,1/n)$$.
If $$\theta = 0$$, then

$$
\mathbb{P}(J_{n}=0)
=\mathbb{P}(|\hat{\theta}|<\sqrt{2}n^{-1/2})
=\mathbb{P}(\left|Z\right|<\sqrt{2})
=2\Phi(\sqrt{2})-1\approx0.8427.
$$

If $$\theta\neq0$$, then

$$
\begin{multline*}
\mathbb{P}(J_{n}=0)
=\mathbb{P}(|\hat{\theta}|<\sqrt{2}n^{-1/2})
=\mathbb{P}(|Zn^{-1/2}+\theta|<\sqrt{2}n^{-1/2})\\
=\mathbb{P}(-\sqrt{2}-\theta\sqrt{n}<Z<\sqrt{2}-\theta\sqrt{n})
=\Phi(\sqrt{2}-\theta\sqrt{n})-\Phi(-\sqrt{2}-\theta\sqrt{n})
\rightarrow 0.
\end{multline*}
$$

### b)

Let $$\mu=\hat{\theta}I_{\{J_{n}=1\}}$$ so that

$$
\hat{f}_{n}(x)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{\left(x-\mu\right)^{2}}{2}\right).
$$

Let $$Z\sim N(0,1)$$.
The KL distance between $$\phi_{0}$$ and $$\hat{f}_{n}$$ is

$$
\begin{align*}
D(\phi_{0},\hat{f}_{n}) & =\int\phi_{0}(z)\left(\log\phi_{0}(z)-\log\hat{f}_{n}(z)\right)dz\\
 & =\mathbb{E}\left[\log\phi_{0}(Z)-\log\hat{f}_{n}(Z)\right]\\
 & =\frac{1}{2}\mathbb{E}\left[-Z^{2}+\left(Z-\mu\right)^{2}\right]\\
 & =\frac{1}{2}\mathbb{E}\left[-2\mu Z+\mu^{2}\right]=\frac{1}{2}\mu^{2}.
\end{align*}
$$

If $$\theta=0$$, this quantity converges to zero in probability since

$$
\mathbb{P}(\mu^{2}>\epsilon)=\mathbb{P}(\hat{\theta}^{2}I_{\{J_{n}=1\}}>\epsilon)\leq\mathbb{P}(\hat{\theta}^{2}>\epsilon)=\mathbb{P}(|Z|>\sqrt{n\epsilon}).
$$

Next, the KL distance between $$\phi_{\hat{\theta}}$$ and $$\hat{f}_{n}$$ is

$$
\begin{align*}
D(\phi_{\hat{\theta}},\hat{f}_{n}) & =\int\phi_{\hat{\theta}}(x)\left(\log\phi_{\hat{\theta}}(x)-\log\hat{f}_{n}(x)\right)dx\\
 & =\int\phi_{0}(z)\left(\log\phi_{0}(z)-\log\hat{f}_{n}(z+\hat{\theta})\right)dz\\
 & =\mathbb{E}\left[\log\phi_{0}(Z)-\log\hat{f}_{n}(Z+\hat{\theta})\right]\\
 & =\frac{1}{2}\mathbb{E}\left[-Z^{2}+\left(Z+\hat{\theta}-\mu\right)^{2}\right]\\
 & =\frac{1}{2}\mathbb{E}\left[2\left(\hat{\theta}-\mu\right)Z+\hat{\theta}^{2}-2\hat{\theta}\mu+\mu^{2}\right]\\
 & =\frac{1}{2}\left(\hat{\theta}^{2}-2\hat{\theta}\mu+\mu^{2}\right).
\end{align*}
$$

By the LLN, $$\hat{\theta}$$ converges to $$\theta$$ in probability.
Suppose that $$\theta\neq0$$.
Our findings in Part (a) imply that $$I_{\{J_{n}=1\}}$$ converges to one in probability.
Therefore, by Theorem 5.5, $$\mu$$ converges to $$\theta$$ in probability and hence $$D(\phi_{\hat{\theta}},\hat{f}_{n})$$ converges to zero in probability.

### c)

Noting that the only difference between the AIC and BIC criteria is replacing the penalty of $$2$$ by $$\log n$$, we can conclude that if $$\theta=0$$, then

$$
\mathbb{P}(J_{n}=0)=2\Phi(\sqrt{\log n})-1\rightarrow1.
$$

Recall that even in the limit, the corresponding quantity for AIC was not one.
Similarly, if $$\theta\neq0$$, then

$$
\mathbb{P}(J_{n}=0)=\Phi(\sqrt{\log n}-\theta\sqrt{n})-\Phi(-\sqrt{\log n}-\theta\sqrt{n})\rightarrow0.
$$

The limiting KL distances are also as before.

## 10.

### a)

Suppose $$\epsilon\sim N(0,\sigma^{2})$$.
Since $$\epsilon$$ is independent of $$\hat{\theta}$$ (recall that $$X_{*}$$ correspond to a sample that hasn't been trained on),

$$
\frac{Y_{*}-\hat{Y}_{*}}{s}=-\frac{\hat{\theta}-\theta}{s}+\frac{\epsilon}{s}\approx N\biggl(0,1+\frac{\sigma^{2}}{s^2}\biggr).
$$

### b)

Similarly to Part (a),

$$
\begin{multline*}
\frac{Y_{*}-\hat{Y}_{*}}{\xi_{n}}=-\frac{\hat{\theta}-\theta}{\xi_{n}}+\frac{\epsilon}{\xi_{n}}=-\frac{\hat{\theta}-\theta}{s}\frac{s}{\sqrt{s^{2}+\sigma^{2}}}+\frac{\epsilon}{\sqrt{s^{2}+\sigma^{2}}}\\
\approx N\biggl(0,\frac{s^{2}}{s^{2}+\sigma^{2}}\biggr)+N\biggl(0,\frac{\sigma^{2}}{s^{2}+\sigma^{2}}\biggr)=N(0,1).
\end{multline*}
$$

## 11.

TODO (Computer experiment).
