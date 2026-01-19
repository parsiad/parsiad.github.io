# %% [raw]
# +++
# aliases = [
#   "/blog/2024/r_squared_and_multiple_regression"
# ]
# date = 2024-09-02
# title = "R squared and multiple regression"
# +++

# %% tags=["no_cell"]
import numpy as np
import statsmodels.api as sm

from _boilerplate import init

init()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Introduction and Results
#
# The *R squared* of a predictor $\hat{Y}$ relative to a target $Y$ is the proportion of the variation in the target that is explained by that predictor.
# In this short note, we introduce the R squared in its most general form.
#
# We then turn our attention to the predictor $\hat{Y}_{\mathrm{ols}} = \hat{\beta}_0 + X^\intercal \hat{\beta}_1$ constructed by [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) (OLS).
# In this case, we prove that the R squared satisfies
#
# $$
# \boxed{R^{2}(Y,\hat{Y}\_{\mathrm{ols}})
# =\frac{\operatorname{Var}(\hat{Y}\_{\mathrm{ols}})}{\operatorname{Var}(Y)}
# =\frac{\operatorname{Cov}(Y,\hat{Y}\_{\mathrm{ols}})}{\operatorname{Var}(Y)}
# =\frac{\operatorname{Cov}(X,Y)^\intercal \hat{\beta}_1}{\operatorname{Var}(Y)}
# =\operatorname{Corr}(Y,\hat{Y}\_{\mathrm{ols}})^{2}}
# $$
#
# where the covariance $\operatorname{Cov}(X, Y)$ is the vector whose entries are $\operatorname{Cov}(X_i, Y)$.
#
# In addition, when the covariance matrix $\operatorname{Cov}(X, X)$ is nonsingular, we can substitute the unique representation for $\hat{\beta}_1$ to get
#
# $$
# \boxed{R^{2}(Y,\hat{Y}_{\mathrm{ols}})
# =\frac{\operatorname{Cov}(X,Y)^\intercal\operatorname{Cov}(X,X)^{-1}\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}.}
# $$
#
# In the case of a one dimensional $X$, the above is the square of the [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between $X$ and $Y$.
#
# These identities may be folklore in the following sense: I believe they are well-known but that a proof, outside of the one dimensional case, is hard to find.
# Note that in the one dimensional case, OLS (with the intercept term $\hat{\beta}_0$) is simply called [simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression).
#
# ## R squared
#
# As mentioned above, the R squared is a fraction of variation.
# A natural notion of variation between two random quantities is the mean squared error:
#
# **Definition (Mean squared error).**
# The *mean squared error* (MSE) between real random variables $A$ and $B$ is
#
# $$
# \operatorname{MSE}(A, B) \equiv \mathbb{E}[(A - B)^2].
# $$
#
# We are now ready to define R squared:
#
# **Definition (R squared).**
# Let $Y$ and $\hat{Y}$ be real random variables. The *R squared* of $\hat{Y}$ relative to $Y$ is
#
# $$
# R^2(Y, \hat{Y})
# \equiv 1 - \frac{\operatorname{MSE}(Y,\hat{Y})}{\operatorname{Var}(Y)}
# $$
#
# Written in the above form, the R squared has an intuitive definition: it is equal to one minus the fraction of unexplained variation (note that the variance itself is a special case of the MSE since $\operatorname{Var}(Y) = \operatorname{MSE}(Y,\mathbb{E}Y)$).
#
# The following facts can be verified by direct substitution into the definition of R squared:
#
# **Fact.**
# *A "perfect" prediction (i.e., $\operatorname{MSE}(Y,\hat{Y})=0$) has an R squared of one.*
#
# **Fact.**
# *A prediction of the mean (i.e., $\hat{Y}=\mathbb{E}Y)$ has an R squared of zero.*
#
# The above give us an upper bound (one) and a "weak" lower bound (zero) for the R squared.
# The lower bound is weak in the sense that it is possible to produce predictors that have a negative R squared but that such predictors are typically pathological as they are worse (in the sense of R squared) than simply predicting the mean.
#
# ## Ordinary least squares
#
# Below, we define OLS with an intercept term.
#
# **Definition (Ordinary least squares).**
# Given a real random variable $Y$ and a real random vector $X$, let
#
# $$
# \hat{Y}\_{\mathrm{ols}}\equiv\hat{\beta}\_{0}+X^{\intercal}\hat{\beta}\_{1}
# $$
#
# be the *ordinary least squares* (OLS) predictor where
#
# $$
# \begin{aligned}
# \hat{\beta}\_{0} & =\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}\_{1}\nonumber \\\\
# \mathbb{E}[XX^{\intercal}]\hat{\beta}\_{1} & =\mathbb{E}[XY]-\mathbb{E}X\hat{\beta}\_{0}.
# \end{aligned}
# $$
#
# *Remark*.
# We could have also defined $\hat{\beta}_0$ and $\hat{\beta}_1$ by the equivalent system
#
# $$
# \begin{aligned}
# \hat{\beta}\_{0} & =\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}\_{1}\nonumber \\\\
# \operatorname{Cov}(X, X)\hat{\beta}\_{1} & =\operatorname{Cov}(X, Y).
# \end{aligned}
# $$
#
# *Remark*.
# Thus far, we have only used a single probability measure: that which is implied by the expectation $\mathbb{E}$.
# In data-driven applications, it is standard practice to fit the coefficients on a subset of data (also known as the [training set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Training_data_set)) while "holding out" the remainder of the data (also known as the [test set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets#Test_data_set)) to evaluate the quality of the fit.
# This results in two distinct expectations: $\mathbb{E}$ and $\mathbb{E}_H$ ($H$ for "hold out").
# In the context of R squared, this results in two natural quantities: $R^2$ and $R_H^2$ where the latter is defined by replacing all expectations $\mathbb{E}$ in the MSE and variance by $\mathbb{E}_H$.
# We stress that the results of this section apply only to the former.
# Put more succinctly, **on the test set, the R squared is not to satisfy the identities** listed at the beginning of this article.
#
# To establish that on the training set, the R squared satisfies the identities listed at the beginning of this article, we use a series of smaller results:
#
# **Lemma 1.**
# *The OLS predictor is unbiased.*
#
# *Proof*.
# Direct computation yields
#
# $$
# \mathbb{E}\hat{Y}\_{\mathrm{ols}}\equiv\hat{\beta}\_{0}+\mathbb{E}X^{\intercal}\hat{\beta}\_{1}=\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}\_{1}+\mathbb{E}X^{\intercal}\hat{\beta}\_{1}=\mathbb{E}Y.\blacksquare
# $$
#
# **Lemma 2.**
# *The second moment of the OLS predictor satisfies*
#
# $$
# \mathbb{E}[\hat{Y}\_{\mathrm{ols}}^{2}]
# = \mathbb{E}[Y\hat{Y}\_{\mathrm{ols}}]
# = \left(\mathbb{E}Y\right)^2 + \operatorname{Cov}(X, Y)^\intercal \hat{\beta}_1.
# $$
#
# *Proof*.
# Direct computation along with the definitions of $\hat{\beta}_0$ and $\mathbb{E}[XX^{\intercal}]\hat{\beta}_1$ reveal
#
# $$
# \begin{aligned}
# \mathbb{E}[Y\hat{Y}\_{\mathrm{ols}}] & =\mathbb{E}\left[Y\left(\hat{\beta}\_{0}+X^{\intercal}\hat{\beta}\_{1}\right)\right]\\\\
#  & =\mathbb{E}\left[Y\hat{\beta}\_{0}+X^{\intercal}Y\hat{\beta}\_{1}\right]\\\\
#  & =\mathbb{E}Y\hat{\beta}\_{0}+\mathbb{E}[X^{\intercal}Y]\hat{\beta}\_{1}\\\\
#  & =\mathbb{E}Y\left(\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}\_{1}\right)+\mathbb{E}[X^{\intercal}Y]\hat{\beta}\_{1}\\\\
#  & =\left(\mathbb{E}Y\right)^{2}-\mathbb{E}X^{\intercal}\mathbb{E}Y\hat{\beta}\_{1}+\mathbb{E}[X^{\intercal}Y]\hat{\beta}\_{1}
# \end{aligned}
# $$
#
# and
#
# $$
# \begin{aligned}
# \mathbb{E}[\hat{Y}\_{\mathrm{ols}}^{2}] & =\mathbb{E}\left[\left(\hat{\beta}\_{0}+X^{\intercal}\hat{\beta}\_{1}\right)^{2}\right]\\\\
#  & =\mathbb{E}\left[\hat{\beta}\_{0}^{2}+2\hat{\beta}\_{0}X^{\intercal}\hat{\beta}\_{1}+\hat{\beta}\_{1}^{\intercal}XX^{\intercal}\hat{\beta}\_{1}\right]\\\\
#  & =\hat{\beta}\_{0}^{2}+2\hat{\beta}\_{0}\mathbb{E}X^{\intercal}\hat{\beta}\_{1}+\hat{\beta}\_{1}^{\intercal}\mathbb{E}[XX^{\intercal}]\hat{\beta}\_{1}\\\\
#  & =\hat{\beta}\_{0}^{2}+2\hat{\beta}\_{0}\mathbb{E}X^{\intercal}\hat{\beta}\_{1}+\left(\mathbb{E}[X^{\intercal}Y]-\hat{\beta}\_{0}\mathbb{E}X^{\intercal}\right)\hat{\beta}\_{1}\\\\
#  & =\left(\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}\_{1}\right)^{2}+\left(\mathbb{E}Y-\mathbb{E}X^{\intercal}\hat{\beta}\_{1}\right)\mathbb{E}X^{\intercal}\hat{\beta}\_{1}+\mathbb{E}\left[X^{\intercal}Y\right]\hat{\beta}\_{1}\\\\
#  & =\left(\mathbb{E}Y\right)^{2}-\mathbb{E}X^{\intercal}\mathbb{E}Y\hat{\beta}\_{1}+\mathbb{E}[X^{\intercal}Y]\hat{\beta}\_{1}
# \end{aligned}
# $$
#
# as desired. $\blacksquare$
#
# **Corollary 3.**
# *The variance of the OLS predictor satisfies*
#
# $$
# \operatorname{Var}(\hat{Y}\_{\mathrm{ols}})
# =\operatorname{Cov}(Y,\hat{Y}\_{\mathrm{ols}}).
# $$
#
# *Proof*.
# Applying Lemmas 1 and 2,
#
# $$
# \operatorname{Cov}(Y,\hat{Y}\_{\mathrm{ols}})=\mathbb{E}[Y\hat{Y}\_{\mathrm{ols}}]-\mathbb{E}Y\mathbb{E}\hat{Y}\_{\mathrm{ols}}=\mathbb{E}[\hat{Y}\_{\mathrm{ols}}^{2}]-\left(\mathbb{E}\hat{Y}\_{\mathrm{ols}}\right)^{2}=\operatorname{Var}(\hat{Y}\_{\mathrm{ols}}).\blacksquare
# $$
#
# **Corollary 4.**
# *The MSE of the OLS predictor can be decomposed as*
#
# $$
# \operatorname{MSE}(Y,\hat{Y}\_{\mathrm{ols}})
# =\operatorname{Var}(Y)-\operatorname{Var}(\hat{Y}\_{\mathrm{ols}}).
# $$
#
# *Proof*.
# Applying Lemmas 1 and 2 and Corollary 3,
#
# $$
# \begin{aligned}
# \operatorname{Var}(Y)-\operatorname{MSE}(Y,\hat{Y}\_{\mathrm{ols}}) & =\mathbb{E}[Y^{2}]-\left(\mathbb{E}Y\right)^{2}-\mathbb{E}[Y^{2}]+2\mathbb{E}[Y\hat{Y}\_{\mathrm{ols}}]-\mathbb{E}[\hat{Y}\_{\mathrm{ols}}^{2}]\\\\
#  & =2\mathbb{E}[Y\hat{Y}\_{\mathrm{ols}}]-\left(\mathbb{E}Y\right)^{2}-\mathbb{E}[\hat{Y}\_{\mathrm{ols}}^{2}]\\\\
#  & =\operatorname{Cov}(Y,\hat{Y}\_{\mathrm{ols}})+\mathbb{E}[Y\hat{Y}\_{\mathrm{ols}}]-\mathbb{E}[\hat{Y}\_{\mathrm{ols}}^{2}]\\\\
#  & =\operatorname{Var}(\hat{Y}\_{\mathrm{ols}}). \blacksquare
# \end{aligned}
# $$
#
# Putting all of the above results together, we arrive at the identities mentioned in the **Introduction and Results** section.
#
# ## Synthetic example
#
# We double-check the claims via a synthetic example below:

# %%
np.random.seed(1)

N = 1_000
p = 5

X = np.random.randn(N, p)
Y = np.random.randn(N)

β = sm.OLS(Y, sm.add_constant(X)).fit().params
Yhat = β[0] + X @ β[1:]

R2 = 1.0 - ((Y - Yhat) ** 2).mean() / Y.var()
CovYYhat = np.cov(Y, Yhat, bias=True)[0, 1]
CovXY = np.cov(X, Y, rowvar=False, bias=True)[:-1, -1]
Corr2 = np.corrcoef(Yhat, Y)[0, 1] ** 2

# On the training set, the R squared satisfies the identities
np.testing.assert_allclose(R2, Yhat.var() / Y.var())
np.testing.assert_allclose(R2, CovYYhat / Y.var())
np.testing.assert_allclose(R2, CovXY @ β[1:] / Y.var())
np.testing.assert_allclose(R2, Corr2)

X_H = np.random.randn(N, p)
Y_H = np.random.randn(N)

Yhat_H = β[0] + X @ β[1:]

R2_H = 1.0 - ((Y_H - Yhat_H) ** 2).mean() / Y_H.var()
CovYYhat_H = np.cov(Y_H, Yhat_H, bias=True)[0, 1]
CovXY_H = np.cov(X_H, Y_H, rowvar=False, bias=True)[:-1, -1]
Corr2_H = np.corrcoef(Yhat_H, Y_H)[0, 1] ** 2

# On the test set, the R squared does not satisfy the identities
assert not np.isclose(R2_H, Yhat_H.var() / Y.var())
assert not np.isclose(R2_H, CovYYhat_H / Y_H.var())
assert not np.isclose(R2_H, CovXY_H @ β[1:] / Y_H.var())
assert not np.isclose(R2_H, Corr2_H)
