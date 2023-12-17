---
date: 2023-12-17 12:00:00-0800
layout: post
redirect_from:
  - /blog/2023/logsumexp_trick/
title: The LogSumExp trick
---
## Motivation

The [softmax function](https://en.wikipedia.org/wiki/Softmax_function) $\sigma$ is used to transform a vector in $\mathbb{R}^n$ to a probability vector in a monotonicity-preserving way.
Specifically, if $x_i \leq x_j$, then $\sigma(x)_i \leq \sigma(x)_j$.

The softmax is typically parametrized by a "temperature" parameter $T$ to yield $\sigma_T(x) \equiv \sigma(x / T)$ which
* shifts more probability mass to the largest component of $x$ as the temperature decays to zero and
* distributes the mass more evenly among the components of $x$ as the temperature grows.

More details regarding the temperature can be found in [a previous blog post](https://parsiad.ca/blog/2022/softmax_sensitivity_to_temperature/).

Algebraically, the softmax is defined as

$$
\sigma(x)_i \equiv \frac{\exp(x_i)}{\sum_j \exp(x_j)}.
$$

This quantity is clearly continuous on $\mathbb{R}^n$ and hence finite there.
However, in the presence of floating point computation, computing this quantity naively can result in blow-up:


```python
x = np.array([768, 1024.])
exp_x = np.exp(x)
exp_x / exp_x.sum()
```

    /tmp/ipykernel_117792/4003806838.py:1: RuntimeWarning: overflow encountered in exp
      exp_x = np.exp(x)
    /tmp/ipykernel_117792/4003806838.py:2: RuntimeWarning: invalid value encountered in divide
      exp_x / exp_x.sum()





    array([nan, nan])



The *LogSumExp trick* is a clever way of reformulating this computation so that it is robust to floating point error.

## The LogSumExp trick

First, let $\bar{x} = \max_i x_i$ and note that

$$
\sigma(x)_{i}=\frac{\exp(x_{i}-\bar{x})}{\sum_{j}\exp(x_{j}-\bar{x})}.
$$

Taking logarithms,

$$
\log(\sigma(x)_{i})=x_{i}-\bar{x}-\log\biggl(\sum_{j}\exp(x_{j}-\bar{x})\biggr).
$$

Exponentiating,

$$
\sigma(x)_{i}=\exp\biggr(x_{i}-\bar{x}-\log\biggl(\sum_{j}\exp(x_{j}-\bar{x})\biggr)\biggr).
$$

In particular, note that $x_j - \bar{x}$ is, by construction, nonpositive and hence has a value less than one when exponentiated.


```python
def softmax(x: np.ndarray) -> np.ndarray:
    x_max = x.max(axis=-1, keepdims=True)
    delta = x - x_max
    lse = np.log(np.exp(delta).sum(axis=-1, keepdims=True))
    return np.exp(delta - lse)
```


```python
x = np.array([768, 1024.])
softmax(x)
```




    array([6.61626106e-112, 1.00000000e+000])


