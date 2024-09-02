---
date: 2024-09-02 12:00:00-0500
layout: post
title: R squared and multiple regression
---
## Motivation

The *R squared* of a predictor $\hat{Y}$ relative to a target $Y$ is the proportion of the variation in the target that is explained by that predictor.
In this short note, we introduce the R squared in its most general form.

We then turn our attention to predictors constructed by ordinary least squares (OLS) with an intercept term and prove that in this case, on the training set, R squared is equal to the square of the correlation between the predictor and target.
This fact is well-known but a proof outside of the case of a single independent variable (a.k.a. [simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)) is hard to find.

## R squared

As mentioned above, the R squared is a fraction of variation.
A natural notion of variation between two random quantities is the mean squared error:

**Definition (Mean squared error).**
The *mean squared error* (MSE) between real random variables $A$ and $B$ is

$$
\operatorname{MSE}(A, B) \equiv \mathbb{E}[(A - B)^2].
$$

We are now ready to define R squared:

**Definition (R squared).**
Let $Y$ and $\hat{Y}$ be real random variables. The *R squared* of $\hat{Y}$ relative to $Y$ is

$$
R^2(Y, \hat{Y})
\equiv 1 - \frac{\operatorname{MSE}(Y,\hat{Y})}{\operatorname{Var}(Y)}
$$

Written in the above form, the R squared has an intuitive definition: it is equal to one minus the fraction of unexplained variation (note that the variance itself is a special case of the MSE since $\operatorname{Var}(Y) = \operatorname{MSE}(Y,\mathbb{E}Y)$).

The following facts can be verified by direct substitution into the definition of R squared:

**Fact.**
*A "perfect" prediction (i.e., $\operatorname{MSE}(Y,\hat{Y})=0$) has an R squared of one.*

**Fact.**
*A prediction of the mean (i.e., $\hat{Y}=\mathbb{E}Y)$ has an R squared of zero.*

The above give us an upper bound (one) and a "weak" lower bound (zero) for the R squared.
The lower bound is weak in the sense that it is possible to produce predictors that have a negative R squared but that such predictors are typically pathological as they are worse (in the sense of R squared) than simply predicting the mean.

## Ordinary least squares

Below, we define OLS with an intercept term.

**Definition (Ordinary least squares).**
Given a real random variable $Y$ and a real random vector $X$, let

$$
\hat{Y}_{\mathrm{ols}}\equiv\hat{\beta}_{0}+X^{\intercal}\hat{\beta}_{1}
$$

be the *ordinary least squares* (OLS) predictor where

$$
\begin{align}
\hat{\beta}_{0} & =\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}_{1}\nonumber \\
\mathbb{E}[XX^{\intercal}]\hat{\beta}_{1} & =\mathbb{E}[XY]-\mathbb{E}X\hat{\beta}_{0}.
\end{align}
$$

*Remark*.
Thus far, we have only used a single probability measure: that which is implied by the expectation $\mathbb{E}$.
In data-driven applications, it is standard practice to fit the coefficients on a subset of data (also known as the [training set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Training_data_set)) while "holding out" the remainder of the data (also known as the [test set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Test_data_set)) to evaluate the quality of the fit.
This results in two distinct expectations: $\mathbb{E}$ and $\mathbb{E}_H$ ($H$ for "hold out").
In the context of R squared, this results in two natural quantities: $R^2$ and $R_H^2$ where the latter is defined by replacing all expectations $\mathbb{E}$ in the MSE and variance by $\mathbb{E}_H$.
We stress that the results of this section apply only to the former.
Put more succinctly, **on the test set, R squared is not guaranteed to be equal to the square of the correlation between the predictor and target** for the OLS predictor.

To establish that on the training set, R squared is equal to the square of the correlation between the predictor and target for the OLS predictor, we first prove a series of smaller results:

**Lemma.**
*The target and OLS predictor are equal in expectation.*

*Proof*.
Direct computation yields

$$
\mathbb{E}\hat{Y}_{\mathrm{ols}}\equiv\hat{\beta}_{0}+\mathbb{E}X^{\intercal}\hat{\beta}_{1}=\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}_{1}+\mathbb{E}X^{\intercal}\hat{\beta}_{1}=\mathbb{E}Y.\blacksquare
$$

**Lemma.**
*The second moment of the OLS predictor is equal to the mean of the product of the target and OLS predictor.
Specifically,*

$$
\mathbb{E}[\hat{Y}_{\mathrm{ols}}^{2}]
= \mathbb{E}[Y\hat{Y}_{\mathrm{ols}}]
$$

*Proof*.
First, note that

$$
Y\hat{Y}_{\mathrm{ols}}=Y\left(\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}_{1}+X^{\intercal}\hat{\beta}_{1}\right)
$$

and

$$
\hat{Y}_{\mathrm{ols}}^{2}=\hat{\beta}_{0}^{2}+2\hat{\beta}_{0}X^{\intercal}\hat{\beta}_{1}+\hat{\beta}_{1}^{\intercal}XX^{\intercal}\hat{\beta}_{1}.
$$

Taking expectations of both expressions, we get

$$
\mathbb{E}[Y\hat{Y}_{\mathrm{ols}}]=\left(\mathbb{E}Y\right)^{2}+\left(\mathbb{E}[X^{\intercal}Y]-\mathbb{E}X^{\intercal}\mathbb{E}Y\right)\hat{\beta}_{1}=\left(\mathbb{E}Y\right)^{2}+\operatorname{Cov}(X,Y)^{\intercal}\hat{\beta}_{1}.
$$

and

$$
\mathbb{E}[\hat{Y}_{\mathrm{ols}}^{2}]=\hat{\beta}_{0}^{2}+2\hat{\beta}_{0}\mathbb{E}X^{\intercal}\hat{\beta}_{1}+\hat{\beta}_{1}^{\intercal}\mathbb{E}[XX^{\intercal}]\hat{\beta}_{1}.
$$

Substituting the expressions for $\hat{\beta}_0$ and $\mathbb{E}[XX^{\intercal}]\hat{\beta}_1$ in the definition of OLS into the above and simplifying yields the desired result. $\blacksquare$

**Corollary.**
*The covariance of the target and the OLS predictor is equal to the variance of the OLS predictor.*

*Proof*.
Applying the above lemmas,

$$
\operatorname{Cov}(Y,\hat{Y}_{\mathrm{ols}})=\mathbb{E}[Y\hat{Y}_{\mathrm{ols}}]-\mathbb{E}Y\mathbb{E}\hat{Y}_{\mathrm{ols}}=\mathbb{E}[\hat{Y}_{\mathrm{ols}}^{2}]-\left(\mathbb{E}\hat{Y}_{\mathrm{ols}}\right)^{2}=\operatorname{Var}(\hat{Y}_{\mathrm{ols}}).\blacksquare
$$

We are now ready to prove the main result:

**Proposition.**
*The R squared of the OLS predictor is the square of the correlation between the target and OLS predictor.
Specifically,*

$$
\boxed{R^{2}(Y,\hat{Y}_{\mathrm{ols}})=\frac{\operatorname{Cov}(Y,\hat{Y}_{\mathrm{ols}})}{\operatorname{Var}(Y)}=\frac{\operatorname{Var}(\hat{Y}_{\mathrm{ols}})}{\operatorname{Var}(Y)}=\operatorname{Corr}(Y,\hat{Y}_{\mathrm{ols}})^{2}}
$$

*where $\operatorname{Corr}$ is the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).*

*Proof*.
The first equality follows from a direct application of the above lemmas:

$$
\begin{align*}
\operatorname{Var}(Y)R^{2}(Y,\hat{Y}_{\mathrm{ols}})
& = \operatorname{Var}(Y)-\operatorname{MSE}(Y,\hat{Y}_{\mathrm{ols}}) \\
& = 2\mathbb{E}[Y\hat{Y}_{\mathrm{ols}}]-\left(\mathbb{E}Y\right)^{2}-\mathbb{E}[\hat{Y}_{\mathrm{ols}}^{2}]\\
& = \operatorname{Cov}(Y,\hat{Y}_{\mathrm{ols}})+\mathbb{E}[Y\hat{Y}_{\mathrm{ols}}]-\mathbb{E}[\hat{Y}_{\mathrm{ols}}^{2}]=\operatorname{Cov}(Y,\hat{Y}_{\mathrm{ols}}).
\end{align*}
$$

The remaining equalities follow from the corollary above. $\blacksquare$

## Synthetic example

We double-check the claims via a synthetic example below:


```python
import numpy as np
import statsmodels.api as sm

np.random.seed(1)

N = 1_000
p = 5

X = np.random.randn(N, p)
Y = np.random.randn(N)

β = sm.OLS(Y, sm.add_constant(X)).fit().params
Yhat = β[0] + X @ β[1:]

R2 = 1. - ((Y - Yhat)**2).mean() / Y.var()
Corr2 = np.corrcoef(Yhat, Y)[0, 1]**2

# On the training set, R2 is equal to the correlation squared
np.testing.assert_allclose(R2, Corr2)

X_H = np.random.randn(N, p)
Y_H = np.random.randn(N)

Yhat_H = β[0] + X @ β[1:]

R2_H = 1. - ((Y_H - Yhat_H)**2).mean() / Y_H.var()
Corr2_H = np.corrcoef(Yhat_H, Y_H)[0, 1]**2

# On the test set, R2 is NOT equal to the correlation squared
assert not np.isclose(R2_H, Corr2_H)
```
