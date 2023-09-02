---
date: 2023-07-14 12:00:00-0800
layout: post
title: Updating the Cholesky Decomposition of a Matrix
---
Sometimes, we already have the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) of a given matrix and wish to update it efficiently when that matrix "grows" in size (i.e., through the addition of new rows/columns).
Instead of recomputing the entire Cholesky decomposition from scratch, we can compute it incrementally.
Moreover, this incremental computation can also be strung together to retrieve an iterative algorithm to compute the *full* Cholesky decomposition.

## Algorithm

Let $A_n \equiv (a_{ij})$ be an $n \times n$ positive definite matrix with Cholesky decomposition $L_nL_n^*$.
Next, consider expanding the size of this matrix (while maintaining positive definiteness):

$$
A_{n+1}=\begin{pmatrix}A_{n} & a_{n+1,1:n}^{*}\\
a_{n+1,1:n} & a_{n+1,n+1}
\end{pmatrix}.
$$

The notation $a_{1:n, n+1}$ signifies a column vector with $n$ entries.
Suppose the Cholesky decomposition of $A_{n + 1}$ has the following form:

$$
L_{n+1}=\begin{pmatrix}L_{n} & 0\\
\ell_{n+1,1:n} & \ell_{n+1,n+1}
\end{pmatrix}.
$$

Simple algebra reveals that

$$
L_{n+1}L_{n+1}^{*}=\begin{pmatrix}A_{n} & L_{n}\ell_{n+1,1:n}^{*}\\
\ell_{n+1,1:n}L_{n}^{*} & \ell_{n+1,1:n}\ell_{n+1,1:n}^{*}+\left|\ell_{n+1,n+1}\right|^{2}
\end{pmatrix}.
$$

This reveals that we need to solve the equations

$$
L_n \ell_{n+1,1:n}^* = a_{n+1,1:n}^*
$$

and

$$
\ell_{n+1,n+1} = \sqrt{a_{n+1,n+1} - \Vert \ell_{n+1, 1:n}\Vert^2}
$$

to obtain the updated Cholesky decomposition.
Since the former involves a triangular matrix, it can be solved by forward substitution in $O(n^2)$ floating point operations (FLOPs).
The latter requires $O(n)$ FLOPs due to the norm.

## Implementation


```python
import numpy as np
import scipy.linalg

def update_chol(chol: np.ndarray, new_vec: np.ndarray) -> np.ndarray:
    """Update the Cholesky factorization of a matrix for real inputs."""
    u = new_vec[:-1]
    α = new_vec[-1]
    v = scipy.linalg.solve_triangular(chol, u, lower=True)
    β = np.sqrt(α - v @ v)
    n = chol.shape[0]
    # WARNING: This is not efficient!
    new_chol = chol.copy()
    new_chol = np.pad(new_chol, [(0, 1), (0, 1)])
    new_chol[:-1, :-1] = chol
    new_chol[-1, :-1] = v
    new_chol[n, n] = β
    return new_chol
```


```python
np.random.seed(42)
x = np.random.randn(5, 5)
a = x.T @ x
np.linalg.cholesky(a)
```




    array([[ 1.72643986,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00926244,  1.9510639 ,  0.        ,  0.        ,  0.        ],
           [-0.02770041,  0.34669923,  1.02437592,  0.        ,  0.        ],
           [ 0.10163684,  0.60454141, -0.41500106,  2.91668584,  0.        ],
           [ 0.31988585,  1.66212358, -1.17204427,  1.10508656,  0.39447333]])




```python
chol = np.linalg.cholesky(a[:-1,:-1])
update_chol(chol, a[-1])
```




    array([[ 1.72643986,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.00926244,  1.9510639 ,  0.        ,  0.        ,  0.        ],
           [-0.02770041,  0.34669923,  1.02437592,  0.        ,  0.        ],
           [ 0.10163684,  0.60454141, -0.41500106,  2.91668584,  0.        ],
           [ 0.31988585,  1.66212358, -1.17204427,  1.10508656,  0.39447333]])



## Iterative method

Note that by applying the algorithm iteratively, it can be used to obtain the *full* Cholesky decomposition of a positive definite matrix $A_N \equiv (a_{ij})$.
The base case is $L_1 = (\sqrt{a_{11}})$.
Assuming each square root takes $c$ FLOPs, the total cost is

$$
c + \sum_{n=1}^{N-1} n^2 + n + 1 + c = \frac{1}{3} N^{3} + \left( c + \frac{2}{3} \right) N - 1.
$$

In particular, the leading term shows that this algorithm is roughly half the complexity of Gaussian elimination applied to arbitrary (i.e., not necessarily positive definite) matrices.
