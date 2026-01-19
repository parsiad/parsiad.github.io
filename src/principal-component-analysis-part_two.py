# %% [raw]
# +++
# aliases = [
#   "/blog/2019/principal_component_analysis_part_two"
# ]
# date = 2019-12-25
# title = "Principal component analysis - part two"
# +++

# %% tags=["no_cell"]
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tabulate import tabulate

from _boilerplate import display_fig, init

init()

# %% [markdown]
# ## Introduction
#
# In [the first post in this series](/blog/2019/principal_component_analysis_part_one/), we outlined the motivation and theory behind principal component analysis (PCA), which takes points $x_1, \ldots, x_N$ in a high dimensional space to points in a lower dimensional space while preserving as much of the original variance as possible.
#
# In this follow-up post, we apply principal components regression (PCR), an algorithm which includes PCA as a subroutine, to a small dataset to demonstrate the ideas in practice.
#
# ## Prerequisites
#
# To understand this post, you will need to be familiar with the following concepts:
#
# * PCA (see [the first post in this series](/blog/2019/principal_component_analysis_part_one/))
# * [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)
#
# ## Ordinary least squares
#
# In ordinary least squares (OLS), we want to find a line of best fit between the points $x_1, \ldots, x_N$ and the labels $y_1, \ldots, y_N$.
#
# Denoting by $X$ the matrix whose rows are the points and $y$ the vector whose entries are the labels, the intercept $\alpha$ and slope (a.k.a. gradient) $\beta$ are obtained by minimizing $\Vert \alpha + X \beta - y \Vert$.
# Some [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus) reveals that the minimum is obtained at the values of $\alpha$ and $\beta$ for which
#
# $$
#     N \alpha = y^\intercal e - \beta^\intercal X^\intercal e
# $$
#
# and
#
# $$
#     X^\intercal X \beta = X^\intercal y - \alpha X^\intercal e
# $$
#
# where $e$ is the vector of all ones.
#
# ## Principal components regression
#
# The idea behind PCR is simple: instead of doing OLS on the high dimensional space, we first map the points to a lower dimensional space obtained by PCA and *then* do OLS.
# In more detail, we
#
# 1. pick a positive integer $k < p$,
# 2. construct the matrix $V_k$ whose columns are the first $k$ principal components of $X$,
# 3. compute $Z_k = X V_k$, a matrix whose rows are the original points transformed to a lower dimensional "PCA space", and
# 4. perform OLS to find a line of best fit between the transformed points and $y$.
#
# By the previous section, we know that the minimum is obtained at the values of the intercept $\alpha_k$ and gradient $\beta_k$ for which
#
# $$
#     N \alpha_k = y^\intercal e - \beta_k^\intercal Z_k^\intercal e
# $$
#
# and
#
# $$
#     Z_k^\intercal Z_k \beta_k = Z_k^\intercal y - \alpha_k Z_k^\intercal e
# $$
#
# Once we have solved these equations for $\alpha_k$ and $\beta_k$, we can predict the label $\hat{y}$ corresponding to a new sample $x$ as $\hat{y} = \alpha_k + x^\intercal V_k \beta_k$.
#
# ### Computational considerations
#
# Due to the result below, the linear system involving $\alpha_k$ and $\beta_k$ is a (permuted) [arrowhead matrix](https://en.wikipedia.org/wiki/Arrowhead_matrix).
# As such, the system can be solved efficiently.
#
# **Lemma.** $Z_k^\intercal Z_k = \Sigma_k^2$ where $\Sigma_k$ is the $k \times k$ diagonal matrix whose entries are the first $k$ principal components of $X$ in descending order.
#
# *Proof*.
# Let $v_j$ denote the $j$-th column of $V_k$.
# Since $v_j$ is a principal component of $X$, it is also an eigenvector of $X^\intercal X$ with eigenvalue $\sigma_j^2$, the square of the $j$-th singular value.
# Therefore, the $(i, j)$-th entry of $Z_k^\intercal Z_k$ is
#
# $$
#     (X v_i)^\intercal (X v_j)
#     = v_i^\intercal X^\intercal X v_j
#     = \sigma_j^2 v_i^\intercal v_j
#     = \sigma_j^2 \delta_{ij}.
# $$
#
# where $\delta_{ij}$ is the [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta).
#
# ## The California housing dataset
#
# The [California housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) from \[1\] has 20,640 samples and 8 predictors.
# Below, we load it and split it into training and test sets.

# %%
np.random.seed(1)
X_all, y_all = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

# %% [markdown]
# ### PCR
#
# A PCR model takes only a few lines to create in [scikit-learn](https://scikit-learn.org/stable/):

# %%
pcr = make_pipeline(
    StandardScaler(),
    PCA(),
    LinearRegression(),
).fit(X_train, y_train)

# %% [markdown]
# For each $k \leq p = 8$, we report the [root mean squared error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) on both the training and test sets.
# The function used to do this is given below along with the resulting table.


# %%
def get_rmses(pipeline: Pipeline, train: bool) -> NDArray:
    """Calculates the RMSE of a model using only the first k PCs for each possible value of k.

    Parameters
    ----------
    pipeline
        An sklearn `Pipeline` whose last two components are `PCA` and `LinearRegression`
    train
        Whether to calculate the RMSE on the train or test set

    Returns
    -------
    A list of RMSEs (the first entry corresponds to k=1 and the last to k=p).
    """
    X, y = (X_train, y_train) if train else (X_test, y_test)
    assert isinstance(pipeline[-2], PCA)
    assert isinstance(pipeline[-1], LinearRegression)
    pca = pipeline[:-1]
    Z = pca.transform(X)
    regression = pipeline[-1]
    rmses = np.empty((Z.shape[1],))
    for k in range(1, Z.shape[1] + 1):
        pred = regression.intercept_ + Z[:, :k] @ regression.coef_[:k]  # type: ignore
        rmse = (mean_squared_error(y, pred)) ** 0.5
        rmses[k - 1] = rmse
    return rmses


# %% tags=["no_input"]
ranks = range(1, X_train.shape[1] + 1)  # type: ignore
rmses_train = get_rmses(pcr, train=True)
rmses_test = get_rmses(pcr, train=False)
rows = zip(ranks, rmses_train, rmses_test)
tabulate(
    rows,
    headers=[
        "Rank (k)",
        "Training set RMSE (in $100,000s)",
        "Test set RMSE (in $100,000s)",
    ],
    tablefmt="html",
)

# %% [markdown]
# Both training and test set RMSEs are (roughly) decreasing functions of the rank.
# This suggests that using all 8 predictors does not cause overfitting.
#
# ### Polynomial PCR
#
# One way to reduce the test set RMSE is to capture nonlinear interactions between features.
#
# Consider, as a toy example, a dataset where each sample $x_i$ has only three predictors: $x_i \equiv (a_i, b_i, c_i)$.
# We can replace each sample $x_i$ by a new sample $x_i^\prime \equiv (a_i, b_i, c_i, a_i^2, a_i b_i, a_i c_i, b_i^2, b_i c_i, c_i^2)$.
# In particular, we have added all possible quadratic monomials in $a_i, b_i, c_i$.
# We refer to these as _derived predictors_ as they are predictors derived from the original features.
# Note that derived predictors need not be quadratic, or even monomials; any function of the original predictors is referred to as a derived predictor.
#
# Returning to the dataset, we add all cubic monomials.
# It is reasonable to expect that unlike OLS applied to $X$, OLS applied to the derived matrix $X^\prime$ will overfit.
# We plot the results of PCR below, observing the effects of overfitting at large values of the rank.

# %%
polynomial_pcr = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=3, include_bias=False),
    PCA(),
    LinearRegression(),
).fit(X_train, y_train)


# %% tags=["no_cell"]
def plot_rmses(
    pipeline: Pipeline, name: str, y_lb: float | None = None, y_ub: float | None = None
) -> None:
    fig = plt.figure(constrained_layout=True)
    rmses_train = get_rmses(pipeline, train=True)
    rmses_test = get_rmses(pipeline, train=False)
    plt.plot(np.arange(rmses_train.size) + 1, rmses_train, label="Training set")
    plt.plot(np.arange(rmses_test.size) + 1, rmses_test, label="Test set")
    if y_lb is None:
        y_lb = plt.gca().get_ylim()[0]
    if y_ub is None:
        y_ub = plt.gca().get_ylim()[1]
    plt.vlines(
        rmses_test.argmin() + 1,
        y_lb,
        y_ub,
        color="k",
        linestyle=":",
        linewidth=1,
    )
    plt.legend()
    plt.ylabel("RMSE in \\$100,000s")
    plt.xlabel("Rank $k$")
    plt.title(f"{name} on California Housing Dataset")
    plt.ylim(y_lb, y_ub)
    display_fig(fig)


# %% tags=["no_input"]
plot_rmses(polynomial_pcr, "Polynomial PCR")

# %% [markdown]
# ### Radial basis function PCR
#
# [Radial basis functions](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html) (RBFs) are another way to generate derived predictors.
# Unlike polynomial features, radial basis functions are more well-behaved: they typically do not suffer from the catastrophic overfitting seen in the previous section.
# While describing RBFs is out of the scope of this article, we show the results of RBF PCR below.

# %%
scaler = StandardScaler()
rbf = make_pipeline(scaler, RBFSampler(gamma=1.0, n_components=1_000, random_state=42))
union = FeatureUnion(
    [
        ("original_features", scaler),
        ("rbf_features", rbf),
    ]
)
rbf_pcr = make_pipeline(
    union,
    PCA(),
    LinearRegression(),
).fit(X_train, y_train)

# %% tags=["no_input"]
plot_rmses(rbf_pcr, "RBF PCR", y_ub=0.75)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Bibliography
#
# \[1\] Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.
