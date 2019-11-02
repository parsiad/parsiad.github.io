---
layout: post
title: All of Statistics - Chapter 9 Solutions
date: 2020-07-25 12:00:00-0800
tags: all-of-statistics
---

## 1.

Let $$\hat{F}$$ be the empirical distribution.
The method of moment estimators (MMEs) satisfy 

$$
\begin{align*}
\mathbb{E}_{\hat{F}}X & =\frac{\tilde{\alpha}}{\tilde{\beta}}\\
\mathbb{E}_{\hat{F}}[X^{2}] & =\frac{\tilde{\alpha}\left(\tilde{\alpha}+1\right)}{\tilde{\beta}^{2}}.
\end{align*}
$$

Solving for $$\tilde{\alpha}$$ and $$\tilde{\beta}$$,

$$
\begin{align*}
\tilde{\alpha} & =\frac{\left(\mathbb{E}_{\hat{F}}X\right)^{2}}{\operatorname{Var}_{\hat{F}}(X)}\\
\tilde{\beta} & =\frac{\mathbb{E}_{\hat{F}}X}{\operatorname{Var}_{\hat{F}}(X)}.
\end{align*}
$$

## 2.

### a)

The MMEs satisfy 

$$
\begin{align*}
\mathbb{E}_{\hat{F}}X & =\frac{\tilde{a}+\tilde{b}}{2}\\
\operatorname{Var}_{\hat{F}}(X) & =\frac{1}{12} \left(\tilde{b}-\tilde{a}\right)^{2}.
\end{align*}
$$

Define $$\tilde{c}=\tilde{b}-\tilde{a}$$.
It follows that 

$$
\tilde{c}=2\sqrt{3} \operatorname{Std}_{\hat{F}}(X).
$$

Since $$\tilde{a}+\tilde{b}=2\tilde{a}+\tilde{c}$$ and $$\tilde{a}+\tilde{b}=2\tilde{b}-\tilde{c}$$,

$$
\begin{align*}
\tilde{a} & =\mathbb{E}_{\hat{F}}X-\sqrt{3} \operatorname{Std}_{\hat{F}}(X)\\
\tilde{b} & =\mathbb{E}_{\hat{F}}X+\sqrt{3} \operatorname{Std}_{\hat{F}}(X).
\end{align*}
$$

### b)

The maximum likelihood estimators (MLEs) maximize

$$
\mathcal{L}(a,b)=\prod_{i}f(X_{i};a,b)=\frac{1}{\left(b-a\right)^{n}}\prod_{i}I_{(a,b)}(X_{i}).
$$

The maximum occurs at $$\hat{a}=\min_{i}X_{i}$$ and $$\hat{b}=\max_{i}X_{i}$$.

### c)

By equivariance, the MLE of $$\tau = (a + b) / 2$$ is $$\hat{\tau} = (\hat{a} + \hat{b}) / 2$$.

### d)

| Estimator                       | Mean squared error (MSE) |
| --------------------------------|--------------------------|
| MLE                             |                    0.015 |
| Non-parametric plug-in estimate |                    0.033 |

The MSE of $$\hat{\tau}$$, the MLE of the mean, is computed using the code below.

```python
import numpy as np

a = 1.
b = 3.
n = 10
n_sims = 10**6

samples = np.random.uniform(low=a, high=b, size=[n_sims, n])

a_mle = np.min(samples, axis=1)  # Maximum likelihood estimator of a.
b_mle = np.max(samples, axis=1)  # Maximum likelihood estimator of b.

tau     = (a     + b    ) / 2.   # Mean of Uniform(a, b).
tau_mle = (a_mle + b_mle) / 2.   # Maximum likelihood estimator of the mean.

mse = np.mean((tau_mle - tau)**2)
```

The non-parameteric plug-in estimate of $$\tau$$ is $$\tilde{\tau} = \mathbb{E}_{\hat{F}} X$$.
Since this estimator is unbiased, its MSE is

$$
\operatorname{Var}(\tilde{\tau})=\frac{1}{n}\operatorname{Var}(X_{1})=\frac{1}{12n}\left(b-a\right)^{2}
$$

## 3.

### a)

Let $$Z$$ be a standard normal random variable so that

$$
0.95=\mathbb{P}(X<\tau)=\mathbb{P}(Z<\left(\tau-\mu\right)/\sigma)=F_{Z}(\left(\tau-\mu\right)/\sigma)
$$

and hence

$$
\tau=g(\mu, \sigma)\equiv F_{Z}^{-1}(0.95)\sigma+\mu.
$$

The MLEs of the mean and standard deviation of the original distribution are $$\hat{\mu}=\mathbb{E}_{\hat{F}}X$$ and $$\hat{\sigma}=\operatorname{Std}_{\hat{F}}(X)$$.
By equivariance, the MLE of $$\tau$$ is $$\hat{\tau} = g(\hat{\mu}, \hat{\sigma})$$.

### b)

By the delta method,

$$
\begin{multline*}
\hat{\operatorname{se}}(\hat{\tau})=\sqrt{\nabla g^{\intercal}\hat{J}_{n}\nabla g}=\sqrt{\frac{1}{n}\begin{pmatrix}1\\
F_{Z}^{-1}(0.95)
\end{pmatrix}^{\intercal}\begin{pmatrix}\sigma^{2} & 0\\
0 & \frac{\sigma^{2}}{2}
\end{pmatrix}\begin{pmatrix}1\\
F_{Z}^{-1}(0.95)
\end{pmatrix}}\\
=\sigma\sqrt{\frac{1}{n}+\frac{1}{2n}\left(F_{Z}^{-1}(0.95)\right)^{2}}
\end{multline*}
$$

A $$1 - \alpha$$ confidence interval is $$\hat{\tau} \pm z_{\alpha / 2} \cdot \hat{\operatorname{se}}(\hat{\tau}).$$

### c)

| Estimator                              | SE    |
| ---------------------------------------|-------|
| Delta Method                           | 0.558 |
| Parametric Bootstrap (100,000 samples) | 0.557 |

Code for the delta method and parametric bootstrap are given below.

```python
import numpy as np
from scipy.stats import norm

data = np.array([ 3.23, -2.50,  1.88, -0.68,  4.43,  0.17,
                  1.03, -0.07, -0.01,  0.76,  1.76,  3.18,
                  0.33, -0.31,  0.30, -0.61,  1.52,  5.43,
                  1.54,  2.28,  0.42,  2.33, -1.03,  4.00,
                  0.39])

se_delta_method = np.std(data) \
                * np.sqrt(1. / n_samples * (1. + 0.5 * norm.ppf(0.95)**2))

n_samples = data.size
n_sims = 10**5
samples = np.std(data) * np.random.randn(n_sims, n_samples) + np.mean(data)
tau_mles = np.std(samples, axis=1) * norm.ppf(0.95) + np.mean(samples, axis=1)
se_parametric_boostrap = np.std(tau_mles)
```

## 4.

By Question 2 (b), the MLE of $$\theta$$ is $$Y = \max_i X_i$$.
Its CDF is $$F_Y = F_{X_1}^n$$.
Therefore, for any $$\epsilon > 0$$, $$F_Y(\theta - \epsilon) < 1$$ and $$F_Y(\theta + \epsilon) = 1$$.
It follows that

$$
\begin{multline*}
\mathbb{P}(\left|Y-\theta\right|>\epsilon)=1-\mathbb{P}(\theta-\epsilon\leq Y\leq\theta+\epsilon)=1-F_{Y}(\theta+\epsilon)+F_{Y}(\theta-\epsilon)\\
=F_{Y}(\theta-\epsilon)=F_{X_{1}}(\theta-\epsilon)^{n}
\end{multline*}
$$

Taking a limit in $$n$$ yields the desired result.

## 5.

Since $$\lambda = \mathbb{E} X_1$$, the MME is the sample mean.

The MLE is also the sample mean.
To see this, note that

$$
\log f(X_{i};\lambda)=\log\left(\frac{\lambda^{X_{i}}e^{-\lambda}}{X_{i}!}\right)=X_{i}\log\lambda-\lambda-\log(X_{i}!)
$$

and hence

$$
(s(X_{i};\lambda))\equiv \frac{d}{d\lambda}\left[f(X_{i};\lambda)\right]=\frac{X_{i}}{\lambda}-1.
$$

Therefore, the derivative of the log likelihood is

$$
\frac{d}{d\lambda}\left[\log\mathcal{L}(\lambda)\right]=\sum_{i}\frac{d}{d\lambda}\left[\log f(X_{i};\lambda)\right]=\frac{1}{\lambda}\mathbb{E}_{\hat{F}}X-n.
$$

Setting this to zero and solving for the parameter yields the desired result.

The Fisher information is

$$
I(\lambda)=\operatorname{Var}(s(X_{i};\lambda))=\frac{1}{\lambda}.
$$

## 6.

### a)

$$\hat{\theta}=\mathbb{E}_{\hat{F}}X$$ is the MLE of the mean.
Let $$Z$$ be a standard normal random variable.
Since 

$$
\psi=\mathbb{P}(Y_{1}=1)=\mathbb{P}(X_{1}>0)=\mathbb{P}(Z>-\theta)=F_{Z}(\theta),
$$

the MLE of $$\psi$$ is $$\hat{\psi}=F_{Z}(\hat{\theta})$$ by equivariance.

### b)

An approximate 95% confidence interval (CI) for $$\psi$$ is $$\hat{\psi}\pm 2\hat{\operatorname{se}}(\hat{\psi})$$ where, by the delta method,

$$
\hat{\operatorname{se}}(\hat{\psi})=\left|f_{Z}(\hat{\theta})\right|\operatorname{se}(\hat{\theta})=f_{Z}(\hat{\theta})\frac{1}{\sqrt{n}}
$$

### c)

By the law of large numbers (LLN), $$\tilde{\psi}$$ converges in probability to $$\mathbb{E}Y_{1}=\mathbb{P}(Y_{1}=1)=\psi$$.

### d)

Similarly to Part (b),

$$
\operatorname{se}(\hat{\psi})=f_{Z}(\theta)\frac{1}{\sqrt{n}}.
$$

Moreover,

$$
\operatorname{se}(\tilde{\psi})=\frac{1}{\sqrt{n}}\operatorname{se}(Y_{1})=\sqrt{\frac{\psi\left(1-\psi\right)}{n}}.
$$

Therefore,

$$
\operatorname{ARE}(\tilde{\psi},\hat{\psi})=\frac{\left(f_{Z}(\theta)\right)^{2}}{\psi\left(1-\psi\right)}
$$

It's possible to show that the ARE achieves its maximum value of $$2 / \pi \approx 0.637$$ at $$\theta = 0$$.
Note that this quantity is necessarily less than one due to the asymptotic optimality of the MLE (Theorem 9.23).

### e)

Suppose the distribution is not normal.
Under sufficient regularity, the LLN guarantees the sample mean $$\hat{\theta}$$ to converge, in probability, to the true mean $$\mu$$ of the distribution.
As such, $$\hat{\psi} = F_Z(\hat{\theta})$$ converges, in probability, to $$F_Z(\mu)$$.

## 7.

### a)

The log-likelihood for drug $$i=1,2$$ is

$$
s(X_{i};p_{i})=\log f(X_{i};p_{i})=\log\binom{n_{i}}{X_{i}}+X_{i}\log p_{i}+\left(n-X_{i}\right)\log\left(1-p_{i}\right).
$$

Taking derivatives,

$$
\frac{d}{dp_{i}}\left[s(X_{i};p_{i})\right]=\frac{X_{i}}{p_{i}}+\frac{X_{i}-n}{1-p_{i}}=\frac{X_{i}-np_{i}}{p_{i}\left(1-p_{i}\right)}.
$$

It follows that the MLE is $$\hat{p}_{i}=X_{i}/n$$.
By equivariance the MLE of $$\psi=p_{1}-p_{2}$$ is $$\hat{\psi}=\hat{p}_{1}-\hat{p}_{2}$$.

### b)

The Fisher information of drug $$i=1,2$$ is

$$
\operatorname{Var}(s(X_{i};p_{i}))=\operatorname{Var}\left(\frac{X_{i}}{p_{i}}+\frac{X_{i}-n_{i}}{1-p_{i}}\right)=\frac{n_{i}}{p_{i}\left(1-p_{i}\right)}.
$$

Since the two trials are independent, the complete Fisher information is

$$
I(p_{1},p_{2})=\operatorname{diag}\left(\frac{n_{1}}{p_{1}\left(1-p_{1}\right)},\frac{n_{2}}{p_{2}\left(1-p_{2}\right)}\right).
$$

### c)

By the delta method,

$$
\hat{\operatorname{se}}(\hat{\psi})=\sqrt{\begin{pmatrix}+1\\
-1
\end{pmatrix}^{\intercal}I(\hat{p}_{1},\hat{p}_{2})^{-1}\begin{pmatrix}+1\\
-1
\end{pmatrix}}=\sqrt{\frac{\hat{p}_{1}\left(1-\hat{p}_{1}\right)}{n_{1}}+\frac{\hat{p}_{2}\left(1-\hat{p}_{2}\right)}{n_{2}}}.
$$

### d)

| Method                                     | 90% CI Lower bound | 90% CI Upper bound |
|--------------------------------------------|--------------------|--------------------|
| Delta Method                               |             -0.009 |              0.129 |
| Parametric Bootstrap (100,000 samples)     |             -0.009 |              0.129 |

```python
import numpy as np
from scipy.stats import norm

n      = 200
x1     = 160
x2     = 148
n_sims = 10**5

p1_mle   = x1/n
p2_mle   = x2/n
psi_mle  = p1_mle - p2_mle
ppf_0p95 = norm.ppf(0.95)

se_delta = np.sqrt(p1_mle * (1. - p1_mle) / n + p2_mle * (1. - p2_mle) / n)
print('90% CI delta method: [{:.3f}, {:.3f}]'.format(
    psi_mle - 1.645 * se_delta_method, psi_mle + 1.645 * se_delta_method))

samples1 = np.random.binomial(n, p1_mle, size=[n_sims])
samples2 = np.random.binomial(n, p2_mle, size=[n_sims])
psi_mles = samples1/n - samples2/n
se_parametric_bootstrap = np.std(psi_mles)
print('90% CI parametric bootstrap: [{:.3f}, {:.3f}]'.format(
    psi_mle - 1.645 * se_parametric_bootstrap,
    psi_mle + 1.645 * se_parametric_bootstrap))
```

## 8.

The log likelihood is

$$
\mathcal{L}(\mu,\sigma)=\sum_{i}\log f(X_{i};\mu,\sigma)=-\frac{n}{2}\log(2\pi)-n\log(\sigma)-\frac{1}{2\sigma^{2}}\sum_{i}\left(X_{i}-\mu\right)^{2}.
$$

Taking derivatives,

$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial\mu} & =\frac{\sum_{i}\left(X_{i}-\mu\right)}{\sigma^{2}}\\
\frac{\partial^{2}\mathcal{L}}{\partial\mu^{2}} & =-\frac{n}{\sigma^{2}}\\
\frac{\partial\mathcal{L}}{\partial\sigma} & =-\frac{n}{\sigma}+\frac{1}{\sigma^{3}}\sum_{i}\left(X_{i}-\mu\right)^{2}\\
\frac{\partial\mathcal{L}}{\partial\mu\partial\sigma} & =-\frac{2\sum_{i}\left(X_{i}-\mu\right)}{\sigma^{2}}\\
\frac{\partial^{2}\mathcal{L}}{\partial\sigma^{2}} & =\frac{n}{\sigma^{2}}-\frac{3}{\sigma^{4}}\sum_{i}\left(X_{i}-\mu\right)^{2}.
\end{align*}
$$

Taking expectations,

$$
\begin{align*}
\mathbb{E}\left[\frac{\partial\mathcal{L}}{\partial\mu\partial\sigma}\right] & =0\\
\mathbb{E}\left[\frac{\partial^{2}\mathcal{L}}{\partial\sigma^{2}}\right] & =-\frac{2n}{\sigma^{2}}.
\end{align*}
$$

Therefore, the Fisher information is

$$
I(\mu,\sigma)=-\frac{n}{\sigma^{2}}\begin{pmatrix}1\\
 & 2
\end{pmatrix}.
$$

## 9.

### a)

Results are given below.

| Method                                     | 95% CI Lower bound | 95% CI Upper bound |
|--------------------------------------------|--------------------|--------------------|
| Delta Method                               |            126.146 |            189.219 |
| Parametric Bootstrap (100,000 samples)     |            126.076 |            189.288 |
| Non-parametric Bootstrap (100,000 samples) |            129.553 |            185.812 |

The MLE of $$\theta$$ is $$\hat{\theta}=g(\hat{\mu})\equiv\exp\hat{\mu}$$ where $$\hat{\mu}=\mathbb{E}_{\hat{F}}X$$ is the MLE of the mean.
By the delta method,

$$
\hat{\operatorname{se}}(\hat{\theta})=\left|g^{\prime}(\hat{\mu})\right|\operatorname{se}(\hat{\mu})=\frac{\exp(\hat{\mu})}{\sqrt{n}}.
$$

Therefore, a 95% CI for $$\theta$$ is $$(1\pm2n^{-1/2})\exp(\hat{\mu})$$.

Code for all three methods is given below.

```python
import numpy as np

np.random.seed(1)
data = np.random.randn(100) + 5.
mu_mle = np.mean(data)

n_sims = 10**5
n_samples = data.size

print('95% CI delta method: [{:.3f}, {:.3f}]'.format(
    (1. - 2. / np.sqrt(n_samples)) * np.exp(mu_mle),
    (1. + 2. / np.sqrt(n_samples)) * np.exp(mu_mle)))

samples = np.random.randn(n_sims, n_samples) + mu_mle
theta_mles = np.exp(np.mean(samples, axis=1))
se_parametric_bootstrap = np.std(theta_mles)
print('95% CI parametric bootstrap: [{:.3f}, {:.3f}]'.format(
    np.exp(mu_mle) - 2. * se_parametric_bootstrap,
    np.exp(mu_mle) + 2. * se_parametric_bootstrap))

indices = np.random.randint(n_samples, size=[n_sims * n_samples])
samples = data[indices]
splits = np.split(samples, n_sims)
theta_mles = np.empty([n_sims])
for i, split in enumerate(splits):
    theta_mles[i] = np.exp(np.mean(split))
se_bootstrap = np.std(theta_mles)
print('95% CI non-parametric bootstrap: [{:.3f}, {:.3f}]'.format(
    np.exp(mu_mle) - 2. * se_bootstrap,
    np.exp(mu_mle) + 2. * se_bootstrap))
```

### b)

![](/assets/img/all-of-statistics-chapter-09/q9b.png)

## 10.

### a)

The CDF of $$\hat{\theta}$$ is $$F(y) = (y / \theta)^n$$.
This, along with CDFs of bootstrap estimators, are plotted below.

![](/assets/img/all-of-statistics-chapter-09/q10a.png)

### b)

The parametric bootstrap estimator of this parameter has a continuous distribution and hence zero probability of being equal to $$\hat{\theta}$$ exactly.
Let $$\hat{\theta}^{*}=\max_{1\leq i\leq n}X_{i}^{*}$$ be the non-parametric bootstrap estimator.
Note that

$$
\mathbb{P}(\hat{\theta}^{*}=\hat{\theta})=1-\mathbb{P}(\hat{\theta}^{*}\neq X_{(n)})=1-\left(1-1/n\right)^{n}\rightarrow1-1/e\approx0.632.
$$

That is, the non-parametric bootstrap estimator has a good chance of being equal to the MLE.
These phenomena are visibile in the plot above.


