# %% [raw]
# +++
# aliases = [
#   "/blog/2015/sdes_and_monte_carlo_in_octave_financial"
# ]
# date = 2015-12-01
# title = "SDEs and Monte Carlo in Octave Financial"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# This post is a tutorial on my fist major contribution as maintainer of [GNU Octave financial package](http://octave.sourceforge.net/financial/): a framework to simulate [stochastic differential equations (SDEs)](https://en.wikipedia.org/wiki/Stochastic_differential_equation) of the form
#
# $$
#     dX_t = F(t, X_t) dt + G(t, X_t) dW_t
# $$
#
# where *W* is a standard *n*-dimensional [Wiener process](https://en.wikipedia.org/wiki/Wiener_process).
#
# To follow along with the examples in this post you'll need a copy of GNU Octave and the Financial package. See the [release announcement post](/blog/2016/octave_financial_0.5.0_released) for installation instructions.
#
# ## A one dimensional example: pricing a European call
#
# A classic problem in finance is that of pricing a European [call option](https://en.wikipedia.org/wiki/Call_option). The price of a European call option with strike price *K* and expiry time *T* years from now is given by
#
# $$
#     \mathbb{E}\left[ e^{-rT} \max\{X_T - K, 0\} \right].
# $$
#
# In the celebrated Black-Scholes framework, the functions *F* and *G* which parameterize the SDE are taken to be *F*(*t*, *X<sub>t</sub>*) = (*r* - 𝛿) *X<sub>t</sub>* and *G*(*t*, *X<sub>t</sub>*) = 𝜎 *X<sub>t</sub>* where *r*, 𝛿, and 𝜎 are the per-annum interest, dividend, and volatility rates of the stock *X*.
# In other words,
#
# $$
#     dX_t = \left(r - \delta\right) X_t dt + \sigma X_t dW_t.
# $$
#
# When *F* and *G* are linear functions of the state variable (as they are in this case), the SDE is called a [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion).
#
# Approximating the above expectation using a sample mean is referred to as *Monte Carlo integration* or *Monte Carlo simulation*.
# Though the Black-Scholes pricing problem happens to be one in which a closed-form solution is known, as an expository example, let's perform Monte Carlo integration to approximate it using an SDE simulation:
#
# ```matlab
# % Test parameters
# X_0 = 100.; K = 100.; r = 0.04; delta = 0.01; sigma = 0.2; T = 1.;
# Simulations = 1e6; Timesteps = 10;
#
# SDE = gbm (r - delta, sigma, "StartState", X_0);
# [Paths, ~, ~] = simByEuler (SDE, 1, "DeltaTime", T, "NTRIALS", Simulations, "NSTEPS", Timesteps, "Antithetic", true);
#
# % Monte Carlo price
# CallPrice_MC = exp (-r * T) * mean (max (Paths(end, 1, :) - K, 0.));
#
# % Compare with the exact answer (Black-Scholes formula): 9.3197
# ```
#
# The `gbm` function is used to generate an object describing geometric Brownian motion (GBM). Under the hood, it invokes the `sde` constructor, which is capable of constructing more general SDEs.
#
# ## Timing
#
# The GNU Octave financial implementation uses broadcasting to speed up computation. Here is a speed comparison of the above with the MATLAB Financial Toolbox, under a varying number of timesteps:
#
# | **Timesteps** | **MATLAB Financial Toolbox (secs.)** | **GNU Octave financial package (secs.)** |
# | ------------- | ------------------------------------ | ---------------------------------------- |
# | 16            | 0.543231                             | 0.048691                                 |
# | 32            | 1.053423                             | 0.064110                                 |
# | 64            | 2.167072                             | 0.097092                                 |
# | 128           | 4.191894                             | 0.162552                                 |
# | 256           | 8.361655                             | 0.294098                                 |
# | 512           | 16.609718                            | 0.568558                                 |
# | 1024          | 32.839757                            | 1.136864                                 |
#
# While both implementations scale more-or-less linearly, the GNU Octave financial package implementation greatly outperforms its MATLAB counterpart.
#
# ## A two dimensional example: pricing a European basket call
#
# Consider now the basket call pricing problem
#
# $$
#     \mathbb{E}\left[ e^{-rT} \max\{\max\{X_T^1, X_T^2\} - K, 0\} \right]
# $$
#
# involving two stocks *X<sup>1</sup>* and *X<sup>2</sup>* which follow the SDEs
#
# $$
#     dX_t^i = \left(r^i - \delta^i\right) X_t^i dt + \sigma^i X_t^i dW_t^i \qquad \text{for } i = 1,2.
# $$
#
# To make matters more interesting, we also assume a correlation between the coordinates of the Wiener process:
#
# $$
#     dW_t^1 dW_t^2 = \rho dt.
# $$
#
# Sample code for this example is below:
#
# ```matlab
# % Test parameters
# X1_0 = 40.; X2_0 = 40.; K = 40.; r = 0.05; delta1 = 0.; delta2 = 0.; sigma1 = 0.5; sigma2 = 0.5; T = 0.25; rho = 0.3;
# Simulations = 1e5; Timesteps = 10;
#
# SDE = gbm ([r-delta1 0.; 0. r-delta2], [sigma1 0.; 0. sigma2], "StartState", [X1_0; X2_0], "Correlation", [1 rho; rho 1]);
# [Paths, ~, ~] = simulate (SDE, 1, "DeltaTime", T, "NTRIALS", Simulations, "NSTEPS", Timesteps, "Antithetic", true);
#
# Max_Xi_T = max (Paths(end, :, :));
# BasketCallPrice_MC = exp (-r * T) * mean (max (Max_Xi_T - K, 0.));
#
# % Compare with the exact answer (Stulz 1982): 6.8477
# ```
