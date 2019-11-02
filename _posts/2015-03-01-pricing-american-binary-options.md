---
layout: post
title: Pricing American binary options
date: 2015-03-01 12:00:00-0800
description: Deriving closed form expressions American binary call and put prices.
---

In this article, we derive the price of an American binary (a.k.a. digital) call and put options assuming that the underlying asset follows geometric Brownian motion. We handle the case in which a finite expiry time is specified for the option. We obtain results for the corresponding perpetual (i.e., no expiry) options by taking limits.

## American binary option

A binary option is a type of option in which pays off either some fixed amount (e.g., one dollar) or nothing at all.

An *American binary put* pays the holder exactly one dollar if the asset on which it is written drops below a specific valuation, called the *strike*. For example, consider owning an American binary put with a strike price of 100 dollars, expiring in a year, and written on AAPL stock (which, for the purposes of this example, we assume trades at 170 dollars today). If at any point in time between now and a year from now AAPL stock drops below 100 dollars, we will receive one dollar.

An *American binary call* is similar, except that it pays the holder exactly one dollar if the asset on which it is written falls below a specific valuation.

## Mathematical formulation

We assume that, under the [pricing measure](https://en.wikipedia.org/wiki/Risk-neutral_measure), the price of the stock (e.g., AAPL) at time *t* is given by

$$S_t = x \exp((r - \delta - \frac{1}{2} \sigma^2)t + \sigma W_t)$$

where the initial price *x* and volatility 𝜎 are positive, the interest rate *r* is real, and the dividend rate 𝛿 is nonnegative. *W* is a standard [Wiener process](https://en.wikipedia.org/wiki/Wiener_process).

Let *K* denote the (nonnegative) strike price. In the case of an American binary put option, if *x* ≤ *K*, then trivially the call option is worth exactly one dollar since the initial price is already below the strike price. For an American binary call, the same is true if *x* ≥ *K*. Therefore, we henceforth assume *x* > *K* for puts and *x* < *K* for calls. In both cases, letting *T* denote the expiry time of the option, its value is

$$\mathbb{E} \left[ \exp(-r  \inf \left\{ t \in [0, T] \colon S_t = K \right\}) \right].$$

The infimum above is a the first [hitting time](https://en.wikipedia.org/wiki/Stopping_time) of the stock to the level *K* before time *T* (we are employing the convention that the infimum of an empty set equals ∞). Substituting the expression for the price of the stock into the infimum, we can transform the above expectation into

$$\mathbb{E} \left[ \exp(-r  \inf \left\{ t \in [0, T] \colon W_t = a + \xi t \right\}) \right].$$

where

$$a = (\log K - \log x) / \sigma \text{ and } \xi = (r - \delta)/\sigma - \sigma/2.$$

The problem has now been transformed into one involving the first hitting time of the Wiener process to a level which varies with time. Call this hitting time 𝜏. Letting *f* denote the density of the random variable 𝜏, we can write the expectation above as

$$\int_0^T e^{-rt} f(t) dt,$$

which we recognize as the [Laplace transform](https://en.wikipedia.org/wiki/Laplace_transform) of $$f \cdot \boldsymbol{1}_{[0,T]}$$ evaluated at *r*.

## Laplace transform

In this section, we determine an expression for the Laplace transform of the density of *f* of the random variable

$$\tau = \inf \left\{ t \in [0, T] \colon W_t = a + \xi t \right\}.$$

We do not assume particular forms for the constants *a* and 𝜉; we only assume that they are real numbers.

By using the [reflection principle](https://en.wikipedia.org/wiki/Reflection_principle_(Wiener_process)), it is possible to derive the density of 𝜏 as

$$f(t) = \frac{\left|a\right|}{\sqrt{2\pi t^3}} \exp \left( - \frac{\left(a - \xi t\right)^2}{2t} \right)$$

(for a detailed derivation, see Shreve, Steven E. *Stochastic calculus for finance II: Continuous-time models*. Vol. 11. Springer Science & Business Media, 2004).

A lengthy but straightforward computation yields the result below.

**Theorem**: Let *r* be real, $$b = \sqrt{\xi^2 + 2r}$$, and 0 < *T* < ∞. If *b* is real, then the Laplace transform of 𝜏 evaluated at *r* is

$$\mathbb{E}\left[e^{-r \tau}\right] = \frac{1}{2} e^{a(\xi - b)} \left\{ 1 + \operatorname{sgn}(a) \operatorname{erf}\left(\frac{bT - a}{\sqrt{2T}}\right) + e^{2ab}\left[1 - \operatorname{sgn}(a) \operatorname{erf}\left(\frac{bT + a}{\sqrt{2T}}\right)\right] \right\}.$$

Note that in the case of the American binary call and put, 𝜉 is chosen such that *b* above is real.

The infinite horizon case (i.e., *T* = ∞) follows from applying the [dominated convergence theorem](https://en.wikipedia.org/wiki/Dominated_convergence_theorem) and taking limits in the above expression:

**Corollary**: Let *r* be real, $$b = \sqrt{\xi^2 + 2r}$$, and *T* = ∞. If *b* is real, then the Laplace transform of 𝜏 evaluated at *r* is

$$\mathbb{E}\left[e^{-r\tau}\right] = e^{a\xi - |a|b}.$$

## Implementation

The MATLAB/GNU Octave code below plots the price of an American put option as a function of the initial stock price *x*. The parameters used are *K* = 100, *r* = 0.04, 𝛿 = 0.01, 𝜎 = 0.01, and *T* = 1.

![](/assets/img/pricing-american-binary-options/price.png)

```matlab
K = 100.;
r = 0.04;
delta = 0.01;
sigma = 0.2;
T = 1.;

x = K:1:K*2;
a = 1. / sigma * log (K ./ x);
xi = (r - delta) / sigma - sigma / 2.;
b = sqrt (xi * xi + 2. * r);

v = 0.5 * exp (a * (xi - b)) .* ( ...
        1 + sign (a) .* erf ((b * T - a) / sqrt (2 * T)) ...
    + exp (2 * a * b) ...
    .* (1 - sign (a) .* erf ((b * T + a) / sqrt (2 * T))) ...
);

plot ([0 1 x], [1 1 v], 'linewidth', 2);
axis ([K/2 K*2 0 1+2^(-5)]);
xlabel ('Initial stock price (x)');
ylabel ('American binary put value (P)');
```

