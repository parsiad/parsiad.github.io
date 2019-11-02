---
layout: post
title: All of Statistics - Chapter 10 Solutions
date: 2020-08-21 12:00:00-0800
tags: all-of-statistics
---

## 1.

Let

$$
Z_{n}=\frac{\hat{\theta}-\theta_{\star}}{\widehat{\operatorname{se}}}.
$$

The probability of correctly rejecting the null hypothesis is

$$
\begin{align}
\beta(\theta_\star)
&=1-\mathbb{P}(\left|W\right|\leq z_{\alpha/2})\\
&=1-\mathbb{P}\left(\frac{\theta_{0}-\theta_{\star}}{\widehat{\operatorname{se}}}-z_{\alpha/2}\leq Z_{n}\leq\frac{\theta_{0}-\theta_{\star}}{\widehat{\operatorname{se}}}+z_{\alpha/2}\right)\\
&=1-\mathbb{P}\left(Z_{n}\leq\frac{\theta_{0}-\theta_{\star}}{\widehat{\operatorname{se}}}+z_{\alpha/2}\right)+\mathbb{P}\left(Z_{n}\leq\frac{\theta_{0}-\theta_{\star}}{\widehat{\operatorname{se}}}-z_{\alpha/2}\right)
\end{align}
$$

If $$Z_{n}$$ is asymptotically normal, taking limits in the above yields expression (10.6).

## 2.

Suppose the conditions of Theorem 10.12 hold and that the CDF $$F$$ of $$T\circ X^{n}$$ is strictly increasing.
Then,

$$
\mathbb{P}_{\theta_{0}}[T(X^{n})\geq T(x^{n})]=1-F[T(x^{n})]
$$

and hence

$$
\begin{align*}
\mathbb{P}_{\theta_{0}}[\text{p-value}\leq y]&=\mathbb{P}_{\theta_{0}}[F(T(x^{n}))\geq1-y]\\
&=1-\mathbb{P}_{\theta_{0}}[T(x^{n})\leq F^{-1}(1-y)]=1-F(F^{-1}(1-y))=y.
\end{align*}
$$



## 3.

Recall that the Wald test rejects if and only if $$|W| > z_{\alpha / 2}$$.
Equivalently, it does not reject if and only if

$$
\hat{\theta} - z_{\alpha / 2} \cdot \widehat{\operatorname{se}}
\leq \theta_0
\leq \hat{\theta} + z_{\alpha / 2} \cdot \widehat{\operatorname{se}}.
$$

## 4.

Note that

$$
\text{p-value}
=\inf\left\{ \sup_{\theta\in\Theta_{0}}\mathbb{P}_{\theta}(T(X^{n})\geq c_{\alpha})\colon T(x^{n})\geq c_{\alpha}\right\}.
$$

Assuming that to each observed test statistic $$T(x^n)$$ there exists a test with $$c_\alpha = T(x^n)$$, the infimum above is attained at $$c_{\alpha}=T(x^{n})$$ and the desired result follows.

## 5.

### a)

The power function is

$$
\beta(\theta)=\mathbb{P}_{\theta}(Y>c)=1-\left(c / \theta\right)^{n}.
$$

### b)

See Part (d).

### c)

We should not reject the null if we observe 0.48 since

$$
\text{p-value}=\mathbb{P}_{1/2}(Y\geq0.48)=1-\left(2\cdot0.48\right)^{20}\approx0.558.
$$

### d)

A test of size $$\alpha$$ is obtained by setting

$$
c_{\alpha}\equiv\frac{1}{2}\left(1-\alpha\right)^{1/n}.
$$

converges to monotonically to 0.5 as $$\alpha$$ converges monotonically to zero from above.
Therefore, all possible tests reject the observation 0.52 (since it is greater than 0.5) and hence the corresponding p-value is exactly zero.
In this case, we can reject the null with zero probability of making a type I error.

## 6.

Let $$\hat{\theta}$$ be the fraction of deaths that occur after passover.
Note that either Wald statistic

$$
W_{0}=\frac{\hat{\theta}-\theta_{0}}{\sqrt{\mathbb{V}_{\theta_{0}}(\hat{\theta})}}=\sqrt{n}\frac{\hat{\theta}-\theta_{0}}{\sqrt{\theta_{0}\left(1-\theta_{0}\right)}}
$$

or

$$
W=\frac{\hat{\theta}-\theta_{0}}{\widehat{\operatorname{se}}(\hat{\theta})}=\sqrt{n}\frac{\hat{\theta}-\theta_{0}}{\sqrt{\hat{\theta}\left(1-\hat{\theta}\right)}}
$$

are asymptotically normal under the null hypothesis and hence appropriate (see Remark 10.5).
The p-value for the latter is

$$
2\Phi(-\left|w\right|)=2\Phi\left(-\left|\sqrt{1919}\frac{997/1919-1/2}{\left(997/1919\right)\left(922/1919\right)}\right|\right)\approx0.087.
$$

This is weak evidence against the null.
A 95% confidence interval for the probability of death before passover is

$$
922/1919 \pm 2 \cdot \widehat{\operatorname{se}} \approx (0.46, 0.50).
$$

## 7.

### a)

Evaluating the code below reveals a p-value of approximately 0.00008 and a 95% confidence interval of approximately (0.01, 0.03).

```python
import numpy as np
import scipy.stats

twain     = [.225, .262, .217, .240, .230, .229, .235, .217]
snodgrass = [.209, .205, .196, .210, .202, .207, .224, .223, .220, .201]
delta_hat = np.mean(twain) - np.mean(snodgrass)
var_hat   = np.var(twain) / len(twain) + np.var(snodgrass) / len(snodgrass)
se_hat    = np.sqrt(var_hat)
wald      = delta_hat / se_hat
p_value   = 2. * scipy.stats.norm.cdf(-np.abs(wald))
ci_95_lo  = delta_hat - 2. * se_hat
ci_95_hi  = delta_hat + 2. * se_hat
```

### b)

The calculations in Part (a) relied on large sample methods despite there being only a handful of samples.
A better choice is a permutation test, which does not require many samples.
Such a test is used below to obtain a p-value of approximately 0.0007.
This is still very strong evidence against the null.

```python
import numpy as np

n_sims = 10**5

def test_stat(data_):
    twain_, snodgrass_ = np.split(data_, [twain.size])
    return np.abs(np.mean(twain_) - np.mean(snodgrass_))

# Compute test statistic on original data.
twain     = [.225, .262, .217, .240, .230, .229, .235, .217]
snodgrass = [.209, .205, .196, .210, .202, .207, .224, .223, .220, .201]
data      = np.concatenate([twain, snodgrass])
observed  = test_stat(data)

# Repeatedly shuffle and compute test statistic.
np.random.seed(1)
perm_stats = np.empty([n_sims])
for i in range(n_sims):
    np.random.shuffle(data)
    perm_stats[i] = test_stat(data)

p_value = np.sum(perm_stats > observed) / n_sims
```

## 8.

### a)

Let $$Z$$ be a standard normal random variable.
Then, under the null hypothesis,

$$
\mathbb{P}_{0}\left(\frac{X_{1}+\cdots+X_{n}}{n}>c\right)=\mathbb{P}\left(Z>c\sqrt{n}\right)=\Phi(-c\sqrt{n}).
$$

Therefore, a test of size $$\alpha$$ is obtained by taking

$$
c=-\frac{\Phi^{-1}(\alpha)}{\sqrt{n}}.
$$

### b)

If the null hypothesis is false, the power is

$$
\beta(1)=\mathbb{P}_{1}\left(\frac{X_{1}+\cdots+X_{n}}{n}>c\right)=\mathbb{P}\left(Z>\left(c-1\right)\sqrt{n}\right)=\Phi(-\left(c-1\right)\sqrt{n}).
$$

### c)

For a fixed size $$\alpha$$,

$$\beta(1) = \Phi(\sqrt{n} + \Phi^{-1}(\alpha)).$$

Taking limits yields the desired result.

## 9.

Let

$$
x_{n}=\frac{\theta_{0}-\theta_{1}}{\widehat{\operatorname{se}}}.
$$

Then,

$$
\begin{multline*}
\beta(\theta_{1})=\mathbb{P}_{\theta_{1}}(\left|Z\right|>z_{\alpha/2})=1-\mathbb{P}_{\theta_{1}}(-z_{\alpha/2}\leq Z\leq z_{\alpha/2})\\
=1-\mathbb{P}_{\theta_{1}}(x_{n}-z_{\alpha/2}\leq Z+x_{n}\leq x_{n}+z_{\alpha/2})=1-\Phi(x_{n}+z_{\alpha/2})+\Phi(x_{n}-z_{\alpha/2}).
\end{multline*}
$$

Since $$x_{n}\rightarrow\operatorname{sign}(\theta_{0}-\theta_{1})\infty$$, it follows that $$\beta(\theta_{1})$$ converges to one in both the $$\theta_{1}>\theta_{0}$$ and $$\theta_{1}<\theta_{0}$$ case.
In other words, as the number of samples increase, the probability of rejection if the null hypothesis is false approaches one.

## 10.

For each of the four weeks, a separate test is performed.
Each test is a paired comparison (Example 10.7) whose null hypothesis is that the rate of death among the two populations is equal.
Evaluating the code below yields

| Week | p-value | Boneferroni corrected p-value |
| ---- | ------- | ----------------------------- |
| -2   | 0.48    | 1                             |
| -1   | 0.0046  | 0.018                         |
|  1   | 0.0068  | 0.027                         |
|  2   | 0.27    | 1                             |

Subject to a Bonferroni correction, there is strong evidence (p-value less than 0.05) to reject the null for weeks -1 and 1.

```python
import numpy as np
import scipy.stats

data = np.array([[55, 141],
                 [33, 145],
                 [70, 139],
                 [49, 161]])

totals = np.sum(data, axis=0)
fracs = data / totals
deltas = fracs @ [1., -1.]
std_errs = np.sqrt(np.sum(fracs * (1. - fracs) / totals, axis=1))
wald_stats = deltas / std_errs
p_values = 2. * scipy.stats.norm.cdf(-np.abs(wald_stats))
bonferroni_p_values = np.minimum(p_values.size * p_values, 1.)
```

## 11.

### a)

| Drug                   | p-value | Odds ratio | Bonferroni p-value |
| ---------------------- | ------- | ---------- | ------------------ |
| Chlorpromazine         | 0.0057  | 0.41       | 0.023              |
| Dimenhydrinate         | 0.52    | 1.2        | 1                  |
| Pentobarbital (100 mg) | 0.63    | 0.85       | 1                  |
| Pentobarbital (150 mg) | 0.01    | 0.56       | 0.4                |

The table above is generated by the code below.

```python
import numpy as np
import scipy.stats

n_patients = np.array([80, 75, 85, 67, 85])
n_nausea = np.array([45, 26, 52, 35, 37])

fracs = n_nausea / n_patients
variances = fracs * (1. - fracs) / n_patients

odds_ratios = fracs[1:] / fracs[0]
deltas = fracs[1:] - fracs[0]
std_errs = np.sqrt(variances[1:] + variances[0])
wald_stats = deltas / std_errs
p_values = 2. * scipy.stats.norm.cdf(-np.abs(wald_stats))

bonferroni_p_values = np.minimum(p_values.size * p_values, 1.)
```

### b)

The Bonferroni p-values are given above.

TODO(BH procedure)

## 12.

### a)

Let $$\hat{\lambda}=n^{-1}\sum_{n}X_{n}$$ be the MLE.
Then, $$\mathbb{V}(\hat{\lambda})=n^{-1}\lambda$$ and hence $$\operatorname{se}(\hat{\lambda})=\sqrt{n^{-1}\lambda}$$.
Therefore, a valid Wald statistic is

$$
W=\sqrt{n}\frac{\hat{\lambda}-\lambda_{0}}{\sqrt{\lambda_{0}}}.
$$

The rejection criteria is $$|W| > c$$.
Taking $$c = z_{\alpha / 2}$$ yields a test that has asymptotic size $$\alpha$$.
Such a rejection region is appropriate when $$n$$ is large.

For small $$n$$, note that

$$
\begin{multline*}
\beta(\lambda_{0})=\mathbb{P}_{\lambda_{0}}(\left|W\right|>c)=1-\mathbb{P}_{\lambda_{0}}\left(\left|\hat{\lambda}-\lambda_{0}\right|\leq c\sqrt{\lambda_0 / n}\right)\\
=1-\mathbb{P}_{\lambda_{0}}(n\lambda_{0}-c\sqrt{n\lambda_{0}}\leq X_1 + \cdots + X_n \leq n\lambda_{0}+c\sqrt{n\lambda_{0}}).
\end{multline*}
$$

Let $$Y=\sum_{i}X_{i}\sim\operatorname{Poisson}(n\lambda_{0})$$.
Then,

$$
\beta(\lambda_{0})=1-F_{Y}((n\lambda_{0}+c\sqrt{n\lambda_{0}})-)+F_{Y}(n\lambda_{0}-c\sqrt{n\lambda_{0}}).
$$

Finding $$c$$ such that this quantity is as close to $$\alpha$$ yields the desired test.

### b)

As discussed in Part (a), $$c$$ is chosen so that the resulting test has power closest to 0.05.
This yields a test of power approximately 0.05572.
Evaluating the code below, a type I error rate of 0.05578 is observed.
If $$n$$ were larger, a Wald test whose power is closer to 0.05 could be constructed.

![](/assets/img/all-of-statistics-chapter-10/q12b.png)

```python
import numpy as np
import scipy.stats

lambda_0 = 1.
n = 20
alpha = 0.05
n_sims = 10**7
c = scipy.stats.norm.ppf(0.975)  # Approximately 1.96.

np.random.seed(1)
samples = np.random.poisson(lam=lambda_0, size=[n_sims, n])
wald = np.sqrt(n / lambda_0) * (np.mean(samples, axis=1) - lambda_0)
n_reject = np.sum(np.abs(wald) > c)
type_one_err_rate = n_reject / n_sims
```

## 13.

Recall that

$$
\log\mathcal{L}=-\frac{n}{2}\log(2\pi)-\log\sigma-\frac{1}{2\sigma^{2}}\sum_{i}\left(X_{i}-\mu\right)^{2}.
$$

Let $$\hat{\mu}=n^{-1}\sum_{i}X_{i}$$ be the MLE.
The likelihood ratio statistic is

$$
\begin{multline*}
\lambda=2\log\mathcal{L}(\hat{\mu})-2\log\mathcal{L}(\mu)=\frac{1}{\sigma^{2}}\left(\sum_{i}\left(X_{i}-\mu_{0}\right)^{2}-\left(X_{i}-\hat{\mu}\right)^{2}\right)\\
=\frac{1}{\sigma^{2}}\left(n\left(\mu_{0}^{2}-\hat{\mu}^{2}\right)-2\left(\mu_{0}-\hat{\mu}\right)\sum_{i}X_{i}\right)=\frac{n}{\sigma^{2}}\left(\mu_{0}^{2}+\hat{\mu}^{2}-2\mu_{0}\hat{\mu}\right)
=n\frac{\left(\hat{\mu}-\mu_{0}\right)^{2}}{\sigma^{2}}.
\end{multline*}
$$

The Wald test statistic is

$$
W=\frac{\hat{\mu}-\mu_{0}}{\operatorname{se}(\hat{\mu})}=\sqrt{n}\frac{\left(\hat{\mu}-\mu_{0}\right)}{\sigma}.
$$

Note, in particular, that $$W^2 = \lambda$$.


## 14.

The likelihood ratio statistic is

$$
\begin{multline*}
\lambda=2\log\mathcal{L}(\hat{\sigma})-2\log\mathcal{L}(\sigma_{0})=2n\left(\log\sigma_{0}-\log\hat{\sigma}\right)+\left(\frac{1}{\sigma_{0}^{2}}-\frac{1}{\hat{\sigma}^{2}}\right)\sum_{i}\left(X_{i}-\mu\right)^{2}\\
=2n\left(\log\sigma_{0}-\log\hat{\sigma}\right)+n\frac{\hat{\sigma}^{2}-\sigma_{0}^{2}}{\sigma_{0}^{2}}.
\end{multline*}
$$

The Wald test statistic is

$$
W=\frac{\hat{\sigma}-\sigma_{0}}{\widehat{\operatorname{se}}(\hat{\sigma})}=\sqrt{n}\frac{\hat{\sigma}-\sigma_{0}}{\sqrt{1/I(\hat{\sigma})}}=\sqrt{2n}\frac{\hat{\sigma}-\sigma_{0}}{\hat{\sigma}}.
$$

It is shown in Question 16 that $$W^{2}/\lambda\xrightarrow{P}1$$ under the null hypothesis.

## 15.

The log likelihood is

$$
\log\mathcal{L}(p)=\log\binom{n}{X}+X\log p+\left(n-X\right)\log(1-p).
$$

Therefore, the likelihood ratio statistic is

$$
\lambda=2X\left(\log\hat{p}-\log p_{0}\right)+2\left(n-X\right)\left(\log(1-\hat{p})-\log(1-p_{0})\right).
$$

The Wald test statistic is

$$
W=\sqrt{n}\frac{\hat{p}-p_{0}}{\sqrt{\hat{p}\left(1-\hat{p}\right)}}.
$$

It is shown in Question 16 that $$W^{2}/\lambda\xrightarrow{P}1$$ under the null hypothesis.

## 16.

Throughout this proof, it is assumed that the density $$f(x;\theta)$$ appearing in the likelihood is sufficiently regular.
A Taylor expansion reveals

$$
\ell(\theta_{0})=\ell(\hat{\theta})+(\hat{\theta}-\theta_{0})\ell^{\prime}(\hat{\theta})+\frac{1}{2}(\hat{\theta}-\theta_{0})^{2}\ell^{\prime\prime}(\hat{\theta})+O((\hat{\theta}-\theta_{0})^{3}).
$$

Note, in particular, that $$\ell^{\prime}(\hat{\theta})=0$$ since $$\hat{\theta}$$ is an MLE.
Therefore,

$$
\lambda=2\log\left(\frac{\mathcal{L}(\hat{\theta})}{\mathcal{L}(\theta_{0})}\right)=-(\hat{\theta}-\theta_{0})^{2}\ell^{\prime\prime}(\hat{\theta})+O((\hat{\theta}-\theta_{0})^{3}).
$$

Moreover,

$$
W^{2}=\frac{(\hat{\theta}-\theta_{0})^{2}}{\widehat{\operatorname{se}}(\hat{\theta})^{2}}=nI(\hat{\theta})(\hat{\theta}-\theta_{0})^{2}.
$$

It follows that

$$
\frac{\lambda}{W^{2}}=\frac{n^{-1}\ell^{\prime\prime}(\hat{\theta})}{-I(\hat{\theta})}+O(\hat{\theta}-\theta_{0}).
$$

Under the null hypothesis, $$\hat{\theta}\xrightarrow{P}\theta_{0}$$.
Therefore, by two applications of Theorem 5.5 (f), $$1/I(\hat{\theta})\rightarrow1/I(\theta_{0})$$ where

$$
I(\theta_{0})=\mathbb{E}_{\theta_{0}}\left[\frac{\partial^{2}\log f(X;\theta_{0})}{\partial\theta^{2}}\right].
$$

Since

$$
\ell^{\prime\prime}(\theta)=\sum_{n}\frac{\partial^{2}\log f(X_{n};\theta)}{\partial\theta^{2}},
$$

by the weak law of large numbers, $$n^{-1}\ell^{\prime\prime}(\hat{\theta})\xrightarrow{P}I(\theta_{0})$$ under the null hypothesis.
The result now follows by Theorem 5.5 (d).

