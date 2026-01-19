# %% [raw]
# +++
# aliases = [
#   "/blog/2018/generating_finite_difference_coefficients"
# ]
# date = 2018-11-02
# title = "Generating finite difference coefficients"
# +++

# %% tags=["no_cell"]
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from _boilerplate import display_fig, init

init()

# %% [markdown]
# Let $u$ be a real-valued $n$-times differentiable function of time.
#
# You are given evaluations of this function $u(t_0), ..., u(t_n)$ at distinct points in time and asked to approximate $u^{(m)}(t)$, the $m$-th derivative of the function evaluated at some given point $t$ ($m \leq n$).
#
# If you have studied numerical methods, you are probably already familiar with how to tackle this problem with what is sometimes referred to as the "method of undetermined coefficients" (or, in equivalent language, by using a [Lagrange interpolating polynomial](https://en.wikipedia.org/wiki/Lagrange_polynomial)). In this post, after reviewing the method, we implement it in a few lines of code.
#
# Consider approximating the derivative $u^{(m)}(t)$ by a linear combination of the observations:
# $$
# \boldsymbol{w}^\intercal \boldsymbol{u}
# \equiv (w_0, \ldots, w_n) (u(t_0), \ldots, u(t_n))^\intercal
# = w_0 u(t_0) + \cdots + w_n u(t_n).
# $$
# Taylor expanding each term around *t*,
# $$
# \boldsymbol{w}^\intercal \boldsymbol{u}
# = \sum_{k = 0}^n w_k \left(
#       u(t) +
#       u^\prime(t) \left(t_k - t\right) +
#       \cdots +
#       u^{(n)}(t) \frac{\left(t_k - t\right)^{n}}{n!} +
#       O(\left(t_k - t\right)^{n+1})
# \right).
# $$
# Rearranging the resulting expression,
# $$
# \boldsymbol{w}^\intercal \boldsymbol{u}
# = O(\max_k \left|t_k - t\right|^{n+1} \Vert \boldsymbol{w} \Vert_\infty) +
# \sum_{k = 0}^{n} u^{(k)}(t) \left(
#       w_0 \frac{\left(t_0 - t\right)^k}{k!} +
#       \cdots +
#       w_n \frac{\left(t_n - t\right)^k}{k!}
# \right).
# $$
#
# The form above makes it clear that the original linear combination is nothing more than an error term (represented in Big O notation) along with a weighted sum of the derivatives of $u$ evaluated at $t$. Since we are interested only in the $m$-th derivative, we would like to pick the weights $\boldsymbol{w}$ such that the coefficient of $u^{(k)}(t)$ is 1 whenever $k=m$ and 0 otherwise. This suggests solving the linear system
#
# $$
# \begin{pmatrix}1 & \cdots & 1\\\\
# (t_{0}-t)^{1\phantom{+1}} & \cdots & (t_{n}-t)^{1\phantom{+1}}\\\\
# (t_{0}-t)^{2\phantom{+1}} & \cdots & (t_{n}-t)^{2\phantom{+1}}\\\\
# \vdots & \ddots & \vdots\\\\
# (t_{0}-t)^{m-1} & \cdots & (t_{n}-t)^{m-1}\\\\
# (t_{0}-t)^{m\phantom{+1}} & \cdots & (t_{n}-t)^{m\phantom{+1}}\\\\
# (t_{0}-t)^{m+1} & \cdots & (t_{n}-t)^{m+1}\\\\
# \vdots & \ddots & \vdots\\\\
# (t_{0}-t)^{n\phantom{+1}} & \cdots & (t_{n}-t)^{n\phantom{+1}}
# \end{pmatrix}\boldsymbol{w}=\begin{pmatrix}0\\\\
# 0\\\\
# 0\\\\
# \vdots\\\\
# 0\\\\
# m!\\\\
# 0\\\\
# \vdots\\\\
# 0
# \end{pmatrix}.
# $$
#
# The matrix on the left hand side is a [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix), and hence this system has a unique solution. Denoting by $\boldsymbol{v}$ the solution of this system, we have
# $$
#     u^{(m)}(t) = \boldsymbol{v}^\intercal \boldsymbol{u} + O(\max_k \left|t_k - t\right|^{n+1} \Vert \boldsymbol{v} \Vert_\infty).
# $$
#
# ## Example application: backward differentiation formula (BDF)
#
# As an application, consider the case in which we want to compute the first derivative of the function ($m=1$) and the observations are made at the points $t_k=t-kh$ where $h$ is some positive constant. In this case, the linear system simplifies significantly:
#
# $$
# \begin{pmatrix}1^{\phantom{n}} & 1^{\phantom{n}} & 1^{\phantom{n}} & 1^{\phantom{n}} & \cdots & 1^{\phantom{n}}\\\\
# 0^{\phantom{n}} & 1^{\phantom{n}} & 2^{\phantom{n}} & 3^{\phantom{n}} & \cdots & n^{\phantom{n}}\\\\
# 0^{\phantom{n}} & 1^{\phantom{n}} & 4^{\boldsymbol{\phantom{n}}} & 9^{\phantom{n}} & \cdots & n^{2}\\\\
# \vdots & \vdots & \vdots & \vdots & \ddots & \vdots\\\\
# 0^{\boldsymbol{\phantom{n}}} & 1^{\phantom{n}} & 2^{n} & 3^{n} & \cdots & n^{n}
# \end{pmatrix}\boldsymbol{w}=\begin{pmatrix}0\\\\
# -1/h\\\\
# 0\\\\
# \vdots\\\\
# 0
# \end{pmatrix}
# $$
#
# Code to solve this linear system for a fixed value of *n* is given below.


# %%
def bdf(n):
    """Creates the coefficient vector for a BDF formula of order `n`.

    Parameters
    ----------
    n: A positive integer.

    Returns
    -------
    A (one-dimensional) numpy array of coefficients `hw`.
    Denoting by `h` a positive step size, the derivative is approximated by
    `(hw[0] * u(t) + hw[1] * u(t-h) + ... + hw[n] * u(t-nh)) / h`
    where u is some real-valued, real-input callable.
    """
    A = np.vander(range(n + 1), increasing=True).transpose()
    b = [(1 - 2 * (k % 2)) * int(k == 1) for k in range(n + 1)]
    return np.linalg.solve(A, b)


# %% [markdown]
# Here are the results of running this code with different values of $n$:

# %% tags=["no_input"]
n = 5
rows = []
for n in range(1, n + 1):
    row = [f"$n={n}$"] + [str(Fraction(x).limit_denominator()) for x in bdf(n)]
    rows.append(row)
tabulate(rows, headers=[""] + [f"$hw_{n}$" for n in range(n + 1)], tablefmt="html")

# %% [markdown]
# As an example of how to read the above table, the third row ($n=3$) tells us
# $$
# h u^\prime(t) = 11/6 \cdot u(t) - 3 \cdot u(t - h) + 3/2 \cdot u(t - 2h) - 1/3 \cdot u(t - 3h) + O(h^4).
# $$
#
# As a crude sanity check, we can verify that the *n*-th BDF formula applied to $e^x$ becomes a better approximation of $e^x$ as *n* increases (recall that the exponential function is its own derivative):

# %%
a = 0.0
b = 10.0
N = 10
h = (b - a) / N
x = np.linspace(a, b, N + 1)
y = np.exp(x)

fig = plt.figure(constrained_layout=True)
plt.semilogy(x, y, label="Exponential function")
for n in range(1, 4):
    approx_exp = (np.convolve(bdf(n), y) / h)[n:-n]
    plt.semilogy(x[n:], approx_exp, "-x", label="BDF {} approximation".format(n))
plt.legend()
display_fig(fig)
