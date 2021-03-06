---
layout: post
title: Brownian bridge
date: 2020-09-27 12:00:00-0800
description: An introduction to the Brownian bridge along with an example application.
---

## Motivation

Given an asset's price at a pair of times (e.g., the price of APPL at 9:00am and 9:30am on a particular trading day), one may be interested in interpolating the price between these times.
Of course, while it is possible to simply interpolate linearly between these times, this approach does not generate a distribution of possible realizations which may be needed for statistical analyses.

Consider, for example, modeling the stock price as a [geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) (GBM).
As the name suggests, the noise from a GBM is generated by a Brownian motion (a.k.a. a [Wiener process](https://en.wikipedia.org/wiki/Wiener_process)).
Since the Brownian bridge describes a Brownian motion between a pair of times at which the value of the Brownian motion has already been realized, it is the appropriate tool to stochastically interpolate the GBM.

## Brownian bridge

As described above, a Brownian bridge $$B$$ is a Wiener process on the time horizon $$[t,T]$$ that is pinned on both sides.
That is, at time $$t$$ it is equal to a known value $$B_{t}$$ and at time $$T$$ it is equal to a known value $$B_{T}$$.
Let $$B_{s}$$ be the Brownian bridge at time $$s$$ in $$(t,T)$$.
It can be shown that $$B_{s}$$ is normal with mean

$$
a+\frac{s-t}{T-t}\left(b-a\right)
$$

and variance

$$
\frac{\left(T-s\right)\left(s-t\right)}{T-t}.
$$

Note that the uncertainty is maximized at exactly the middle of the interval $$(t+T)/2$$.

## Simulation

To simulate the bridge, pick a point $$s$$ in the interval $$(t,T)$$ and use the above distributional characterization to sample $$B_{s}$$.
Now, the values of $$B$$ on the mesh $$\{t,s,T\}$$ are known.

To further refine this mesh, pick a new point $$r$$ that is in either $$(t,s)$$ or $$(s,T)$$.
Depending on which interval is picked, a Brownian bridge with the endpoints $$(B_{t},B_{s})$$ or $$(B_{s},B_{T})$$ can be used to sample $$B_{r}$$.
This procedure can be repeated as many times as necessary to obtain as fine a mesh as desired.

## Application to Geometric Brownian motion

Consider now a geometric Brownian motion

$$
dS_{t}=\mu S_{t}dt+\sigma S_{t}dW_{t}.
$$

Let $$X_{t}=\log S_{t}$$. By Ito's lemma,

$$
dX_{t}=\left(\mu-\frac{1}{2}\sigma^{2}\right)dt+\sigma dW_{t}.
$$

If $$X_{t}=a$$ and $$X_{T}=b$$,

$$
b-a=\left(\mu-\frac{1}{2}\sigma^{2}\right)\left(T-t\right)+\sigma\left(W_{T}-W_{t}\right)
$$

and hence

$$
W_{T}-W_{t}=\frac{b-a-\left(\mu-\frac{1}{2}\sigma^{2}\right)\left(T-t\right)}{\sigma}.
$$

Therefore, simulating the process $$X$$ between the initial and final times is equivalent to simulating a Brownian bridge that is pinned at $$B_{t}=0$$ and $$B_{T}=W_{T}-W_{t}$$ as given above.

## Interpolating the NASDAQ in 2017

*Remark*. Below, the price of a symbol on a particular trading day is defined to be the midpoint between its open and close price.

The price of the NASDAQ on the first and last trading days of 2017 was 5427.35 and 6928.00, respectively.
In a previous article, the maximum likelihood estimators for the NASDAQ drift and volatility (assuming a GBM model) in 2017 were found to be  24.696% and 7.530% per annum, respectively.
With only this information, the ideas from the previous section can be used to generate realizations which agree with the NASDAQ on the first and last trading days of 2017.

*Disclaimer*. The code below is optimized for readability, not speed.

```python
drift = 0.24696
vol = 0.07530

n_trading_days_in_2017 = 251
nasdaq_20170103 = 5427.35
nasdaq_20171229 = 6928.00

dt = 1. / (n_trading_days_in_2017 - 1)
x = np.empty([n_simulations, n_trading_days_in_2017])
x[:,  0] = np.log(nasdaq_20170103)  
x[:, -1] = np.log(nasdaq_20171229)
drift_in_logspace = drift - 0.5 * vol**2
for n in range(1, n_trading_days_in_2017 - 1):
    s = n * dt
    t = s - dt
    T = 1.
    a = x[:, n - 1]
    b = x[:, -1]
    dw = (b - a - drift_in_logspace * (T - t)) / vol
    mean = (s - t) / (T - t) * dw
    var = (T - s) * (s - t) / (T - t)
    samples = mean + np.sqrt(var) * np.random.randn(n_simulations)
    x[:, n] = a + drift_in_logspace * dt + vol * samples

simulations = np.exp(x)
```

The true value of the NASDAQ along with 9 Brownian bridge simulations are shown.

![](/assets/img/brownian-bridge/nasdaq.png)
