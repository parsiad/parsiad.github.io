---
layout: post
title: Principal component analysis - part two
date: 2019-12-25 12:00:00-0800
description: In this follow-up post, we apply principal components regression to a small dataset.
---

## Introduction

In [the first post in this series](/blog/2019/principal-component-analysis-part-one/), we outlined the motivation and theory behind principal component analysis (PCA), which takes points $$x_1, \ldots, x_N$$ in a high dimensional space to points in a lower dimensional space while preserving as much of the original variance as possible.

In this follow-up post, we apply principal components regression (PCR), an algorithm which includes PCA as a subroutine, to a small dataset to demonstrate the ideas in practice.

## Prerequisites

To understand this post, you will need to be familiar with the following concepts:

* PCA (see [the first post in this series](/blog/2019/principal-component-analysis-part-one/))
* [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)

## Ordinary least squares

In ordinary least squares (OLS), we want to find a line of best fit between the points $$x_1, \ldots, x_N$$ and the labels $$y_1, \ldots, y_N$$.

Denoting by $$X$$ the matrix whose rows are the points and $$y$$ the vector whose entries are the labels, the intercept $$\alpha$$ and slope (a.k.a. gradient) $$\beta$$ are obtained by minimizing $$\Vert \alpha + X \beta - y \Vert$$.
Some [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus) reveals that the minimum is obtained at the values of $$\alpha$$ and $$\beta$$ for which

$$
\begin{align*}
    N \alpha & = y^\intercal e - \beta^\intercal X^\intercal e \\
    X^\intercal X \beta & = X^\intercal y - \alpha X^\intercal e
\end{align*}
$$

where $$e$$ is the vector of all ones.

## Principal components regression

The idea behind PCR is simple: instead of doing OLS on the high dimensional space, we first map the points to a lower dimensional space obtained by PCA and *then* do OLS.
In more detail, we

1. pick a positive integer $$k < p$$,
2. construct the matrix $$V_k$$ whose columns are the first $$k$$ principal components of $$X$$,
3. compute $$Z_k = X V_k$$, a matrix whose rows are the original points transformed to a lower dimensional "PCA space", and
4. perform OLS to find a line of best fit between the transformed points and $$y$$.

By the previous section, we know that the minimum is obtained at the values of the intercept $$\alpha_k$$ and gradient $$\beta_k$$ for which

$$
\begin{align*}
    N \alpha_k & = y^\intercal e - \beta_k^\intercal Z_k^\intercal e \\
    Z_k^\intercal Z_k \beta_k & = Z_k^\intercal y - \alpha_k Z_k^\intercal e
\end{align*}
$$

Once we have solved these equations for $$\alpha_k$$ and $$\beta_k$$, we can predict the label $$\hat{y}$$ corresponding to a new sample $$x$$ as $$\hat{y} = \alpha_k + x^\intercal V_k \beta_k$$.

### Computational considerations

Due to the result below, the linear system involving $$\alpha_k$$ and $$\beta_k$$ is a (permuted) [arrowhead matrix](https://en.wikipedia.org/wiki/Arrowhead_matrix).
As such, the system can be solved efficiently.

**Lemma.** $$Z_k^\intercal Z_k = \Sigma_k^2$$ where $$\Sigma_k$$ is the $$k \times k$$ diagonal matrix whose entries are the first $$k$$ principal components of $$X$$ in descending order.

*Proof*.
Let $$v_j$$ denote the $$j$$-th column of $$V_k$$.
Since $$v_j$$ is a principal component of $$X$$, it is also an eigenvector of $$X^\intercal X$$ with eigenvalue $$\sigma_j^2$$, the square of the $$j$$-th singular value.
Therefore, the $$(i, j)$$-th entry of $$Z_k^\intercal Z_k$$ is

$$
\begin{equation}
    (X v_i)^\intercal (X v_j)
    = v_i^\intercal X^\intercal X v_j
    = \sigma_j^2 v_i^\intercal v_j
    = \begin{cases}
    	\sigma_j^2 & \text{if } i = j \\
    	0 & \text{if } i \neq j.
    \end{cases}
\end{equation}
$$

## Boston house prices dataset

The [Boston house prices dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing) from \[1\] has 506 samples and 13 predictors.
For each $$k \leq p = 13$$, we fit using PCR on the first 405 samples (the training set) and report the [root mean squared error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) on both the training set and the set of remaining 101 samples (the test set).

|   Rank (k) |   Training set RMSE (in $1000s) |   Test set RMSE (in $1000s) |
|------------|---------------------------------|-----------------------------|
|          1 |                         7.16734 |                     7.57061 |
|          2 |                         6.7612  |                     6.91805 |
|          3 |                         5.61098 |                     5.80307 |
|          4 |                         5.42897 |                     6.07821 |
|          5 |                         4.89393 |                     5.78428 |
|          6 |                         4.88918 |                     5.76014 |
|          7 |                         4.86875 |                     5.78133 |
|          8 |                         4.82526 |                     5.71379 |
|          9 |                         4.818   |                     5.74823 |
|         10 |                         4.78993 |                     5.73366 |
|         11 |                         4.75929 |                     5.67803 |
|         12 |                         4.6241  |                     5.38402 |
|         13 |                         4.54322 |                     5.32823 |

Both training and test set RMSEs are (roughly) decreasing functions of the rank.
This suggests that using all 13 predictors does not cause overfitting.

Code used to generate the table above is given in the appendix.

### Deriving predictors

One way to reduce the test set RMSE is to introduce more predictors into the model.
Consider, as a toy example, a dataset where each sample $$x_i$$ has only three predictors: $$x_i \equiv (a_i, b_i, c_i)$$.
We can replace each sample $$x_i$$ by a new sample $$x_i^\prime \equiv (a_i, b_i, c_i, a_i^2, a_i b_i, a_i c_i, b_i^2, b_i c_i, c_i^2)$$.
In particular, we have added all possible quadratic monomials in $$a_i, b_i, c_i$$.
These new entries are referred to as "derived" predictors.
Note that derived predictors need not be quadratic, or even monomials; any function of the original predictors is referred to as a derived predictor.

Returning to the Boston house prices dataset, of all possible derived cubic monomial predictors, we randomly choose roughly 100 to add to our dataset.
Since we have approximately 400 training samples, it is reasonable to expect that unlike OLS applied to $$X$$, OLS applied to the derived matrix $$X^\prime$$ will almost certainly overfit.
We plot the results of PCR below, observing the effects of overfitting after approximately a rank of greater than 80.

![](/assets/img/principal-component-analysis-part-two/train_test_plot.png)

## Bibliography

\[1\] Harrison Jr, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of environmental economics and management*, 5(1), 81-102.

## Appendix: code

```python
import numpy as np
from sklearn.datasets import load_boston
from tabulate import tabulate

TRAIN_TEST_SPLIT_FRACTION = 0.2

X, y = load_boston(return_X_y=True)
N, p = X.shape

# Train test split.
np.random.seed(123)
perm = np.random.permutation(N)
X, y = X[perm], y[perm]
N_test = int(TRAIN_TEST_SPLIT_FRACTION * N)
N_train = N - N_test
X_test, X_train = np.split(X, [N_test])
y_test, y_train = np.split(y, [N_test])

# Normalize data.
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_test, X_train = [(X_sub - X_mean) / X_std for X_sub in [X_test, X_train]]

_, _, V_T = np.linalg.svd(X_train)
V = V_T.T

rows = []
for k in range(1, p + 1):
    V_k = V[:, :k]
    Z_k = X_train @ V_k

    # Solve for alpha_k and beta_k by adding a bias column to Z_k.
    # This is not efficient (see "Computational considerations" above).
    Z_k_bias = np.concatenate([np.ones([N_train, 1]), Z_k], axis=1)
    solution = np.linalg.solve(Z_k_bias.T @ Z_k_bias, Z_k_bias.T @ y_train)
    alpha_k = solution[0]
    beta_k = solution[1:]

    V_k_beta_k = V_k @ beta_k

    row = [k]
    for X_sub, y_sub in [(X_train, y_train), (X_test, y_test)]:
        y_hat = alpha_k + X_sub @ V_k_beta_k
        error = y_hat - y_sub
        rmse = np.sqrt(np.mean(error ** 2))
        row.append(rmse)
    rows.append(row)

table = tabulate(rows, headers=['Rank (k)', 'Training set RMSE (in $1000s)',
                                'Test set RMSE (in $1000s)'], tablefmt='github')
print(table)
```
