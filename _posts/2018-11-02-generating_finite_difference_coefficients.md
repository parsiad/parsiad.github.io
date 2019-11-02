---
date: 2018-11-02 12:00:00-0800
layout: post
redirect_from:
  - /blog/2018/generating-finite-difference-coefficients/
title: Generating finite difference coefficients
---
Let $u$ be a real-valued $n$-times differentiable function of time.

You are given evaluations of this function $u(t_0), ..., u(t_n)$ at distinct points in time and asked to approximate $u^{(m)}(t)$, the $m$-th derivative of the function evaluated at some given point $t$ ($m \leq n$).

If you have studied numerical methods, you are probably already familiar with how to tackle this problem with what is sometimes referred to as the "method of undetermined coefficients" (or, in equivalent language, by using a [Lagrange interpolating polynomial](https://en.wikipedia.org/wiki/Lagrange_polynomial)). In this post, after reviewing the method, we implement it in a few lines of code.

Consider approximating the derivative $u^{(m)}(t)$ by a linear combination of the observations:
\begin{equation}
\boldsymbol{w}^\intercal \boldsymbol{u}
\equiv (w_0, \ldots, w_n) (u(t_0), \ldots, u(t_n))^\intercal
= w_0 u(t_0) + \cdots + w_n u(t_n).
\end{equation}
Taylor expanding each term around *t*,
\begin{equation}
\boldsymbol{w}^\intercal \boldsymbol{u}
= \sum_{k = 0}^n w_k \left(
      u(t) +
      u^\prime(t) \left(t_k - t\right) +
      \cdots +
      u^{(n)}(t) \frac{\left(t_k - t\right)^{n}}{n!} +
      O(\left(t_k - t\right)^{n+1})
\right).
\end{equation}
Rearranging the resulting expression,
\begin{equation}
\boldsymbol{w}^\intercal \boldsymbol{u}
= O(\max_k \left|t_k - t\right|^{n+1} \Vert \boldsymbol{w} \Vert_\infty) +
\sum_{k = 0}^{n} u^{(k)}(t) \left(
      w_0 \frac{\left(t_0 - t\right)^k}{k!} +
      \cdots +
      w_n \frac{\left(t_n - t\right)^k}{k!}
\right).
\end{equation}

The form above makes it clear that the original linear combination is nothing more than an error term (represented in Big O notation) along with a weighted sum of the derivatives of $u$ evaluated at $t$. Since we are interested only in the $m$-th derivative, we would like to pick the weights $\boldsymbol{w}$ such that the coefficient of $u^{(k)}(t)$ is 1 whenever $k=m$ and 0 otherwise. This suggests solving the linear system

<img alt="" src="data:image/png;base64,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" />

The matrix on the left hand side is a [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix), and hence this system has a unique solution. Denoting by $\boldsymbol{v}$ the solution of this system, we have
\begin{equation}
    u^{(m)}(t) = \boldsymbol{v}^\intercal \boldsymbol{u} + O(\max_k \left|t_k - t\right|^{n+1} \Vert \boldsymbol{v} \Vert_\infty).
\end{equation}

## Example application: backward differentiation formula (BDF)

As an application, consider the case in which we want to compute the first derivative of the function ($m=1$) and the observations are made at the points $t_k=t-kh$ where $h$ is some positive constant. In this case, the linear system simplifies significantly:

<img alt="" src="data:image/png;base64,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" />

Code to solve this linear system for a fixed value of *n* is given below.


```python
import numpy as np


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
```

Here are the results of running this code with different values of $n$:




<table>
<thead>
<tr><th>     </th><th>$hw_0$  </th><th style="text-align: right;">  $hw_1$</th><th>$hw_2$  </th><th>$hw_3$  </th><th>$hw_4$  </th><th>$hw_5$  </th></tr>
</thead>
<tbody>
<tr><td>$n=1$</td><td>1       </td><td style="text-align: right;">      -1</td><td>        </td><td>        </td><td>        </td><td>        </td></tr>
<tr><td>$n=2$</td><td>3/2     </td><td style="text-align: right;">      -2</td><td>1/2     </td><td>        </td><td>        </td><td>        </td></tr>
<tr><td>$n=3$</td><td>11/6    </td><td style="text-align: right;">      -3</td><td>3/2     </td><td>-1/3    </td><td>        </td><td>        </td></tr>
<tr><td>$n=4$</td><td>25/12   </td><td style="text-align: right;">      -4</td><td>3       </td><td>-4/3    </td><td>1/4     </td><td>        </td></tr>
<tr><td>$n=5$</td><td>137/60  </td><td style="text-align: right;">      -5</td><td>5       </td><td>-10/3   </td><td>5/4     </td><td>-1/5    </td></tr>
</tbody>
</table>



As an example of how to read the above table, the third row ($n=3$) tells us
\begin{equation}
h u^\prime(t) = 11/6 \cdot u(t) - 3 \cdot u(t - h) + 3/2 \cdot u(t - 2h) - 1/3 \cdot u(t - 3h) + O(h^4).
\end{equation}

As a crude sanity check, we can verify that the *n*-th BDF formula applied to $e^x$ becomes a better approximation of $e^x$ as *n* increases (recall that the exponential function is its own derivative):


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


mpl.style.use("fivethirtyeight")
mpl.rcParams["lines.linewidth"] = 2

a = 0.0
b = 10.0
N = 10
h = (b - a) / N
x = np.linspace(a, b, N + 1)
y = np.exp(x)

plt.figure(figsize=(1.618 * 5.0, 5.0))
plt.semilogy(x, y, label="Exponential function")
for n in range(1, 4):
    approx_exp = (np.convolve(bdf(n), y) / h)[n:-n]
    plt.semilogy(x[n:], approx_exp, "-x", label="BDF {} approximation".format(n))

plt.legend()
plt.show()
```


    
![png](/assets/2018-11-02-generating_finite_difference_coefficients_files/2018-11-02-generating_finite_difference_coefficients_17_0.png)
    

