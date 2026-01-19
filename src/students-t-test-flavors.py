# %% [raw]
# +++
# aliases = [
#   "/blog/2024/students_t_test_flavors"
# ]
# date = 2024-12-18
# title = "Student's t-test flavors"
# +++

# %% tags=["no_cell"]
import itertools

import numpy as np
import scipy
from numpy.typing import NDArray
from tabulate import tabulate

from _boilerplate import init

init()

# %% [markdown]
# A Student's t-test is any hypothesis test where the test statistic follows a Student's t-distribution.
# In this short note, we list and distinguish between the most commonly used ones.
#
# Throughout, we denote by $t \mapsto F_\nu(t)$ the CDF of a Student's t-distribution with $\nu$ degrees of freedom.
#
# ## Dependent (a.k.a. paired) t-test
#
# Given pairs $(X_1, Y_1), \ldots, (X_N, Y_N)$, let $D_n = X_n - Y_n$.
# A *paired t-test* assumes
#
# $$
# D_1, \ldots, D_N \sim \mathcal{N}(\mu, \sigma^2) \text{ are i.i.d.}
# $$
#
# and tests the null hypothesis $\mu = 0$.
#
# **Example (Running shoes).**
# A shoe company has just released their newest running shoe, "SpeedBoot v2.0".
# The popular YouTube channel "ShoeScoop" wants to conduct an independent review to see whether this shoe improves (or degrades) running times relative to its predecessor, "SpeedBoot v1.0".
# ShoeScoop hires 10 runners to run the same distance at a fixed level of effort with each shoe (allowing each runner an ample break between their runs).
# Each runner produces a pair of times $X_n$ and $Y_n$.
# Putting these pairs into a paired t-test, ShoeScoop can determine if there is a statistically significant difference in average running times between the versions.
#
# The test statistic is
#
# $$
# T = \sqrt{N} \frac{\overline{D}}{s_D}
# $$
#
# where $\overline{D}$ is the sample mean and $s_D$ is the [Bessel corrected](https://en.wikipedia.org/wiki/Bessel's_correction) sample standard deviation of $D_1, \ldots, D_N$.
# [It can be shown that $T$ is a Student's t-distribution with $\nu = N - 1$ degrees of freedom](https://en.wikipedia.org/wiki/Student%27s_t-distribution#As_the_distribution_of_a_test_statistic).
# Let $t$ denote the *observed* value of the test statistic (distinguishing it from the random variable $T$).
# From [a previous article on hypothesis testing](/blog/2019/hypothesis_testing_for_mathematicians), we know that the p-value corresponding to a test that rejects the null hypothesis when the magnitude of the test statistic is sufficiently large is
#
# $$
# \text{p-value}
# = \mathbb{P}(|T| \geq |t|)
# = 2 \mathbb{P}(T \leq -|t|)
# = 2 F_{N - 1}(-|t|)
# $$
#
# where the second equality is a consequence of the symmetry of the Student's t-distribution.
#
# **Example (Running shoes, cont'd).**
# Paired times (in minutes) for the two versions of SpeedBoot along with the test statistic ($t$) and p-value are below.

# %% tags=["no_input"]
np.random.seed(42)
N = 10
Z = np.random.normal(loc=60.0, scale=15.0, size=N)
X = np.round(Z + np.random.normal(loc=0.0, scale=1.0, size=N), decimals=2)
Y = np.round(
    Z
    + np.random.normal(loc=-1.5, scale=1.0, size=N)
    + np.random.normal(loc=0.0, scale=1.0, size=N),
    decimals=2,
)

tabulate(
    zip(np.arange(N), X, Y),
    headers=(
        "Runner ($n$)",
        "SpeedBoot v1.0 Time ($X_n$)",
        "SpeedBoot v2.0 Time ($Y_n$)",
    ),
    tablefmt="html",
)


# %%
def dep_ttest(X: NDArray, Y: NDArray) -> tuple[float, float]:
    """Dependent (a.k.a. paired) t-test (should give results identical to `scipy.stats.ttest_rel`)."""
    D = X - Y
    (N,) = D.shape
    t = D.mean() / scipy.stats.sem(D)
    pvalue = 2.0 * scipy.stats.t.cdf(-np.abs(t), df=N - 1)
    return t.item(), pvalue.item()


t, pvalue = dep_ttest(X, Y)
print(f"{t=}, {pvalue=}")

# %% editable=true slideshow={"slide_type": ""} tags=["no_cell"]
expected = scipy.stats.ttest_rel(X, Y)
assert np.isclose(expected.statistic, t)
assert np.isclose(expected.pvalue, pvalue)

# %% [markdown]
# ## Independent (a.k.a. two sample) t-test
#
# Given two sets of observations $X_1, \ldots, X_N$ and $Y_1, \ldots, Y_M$, an *independent t-test* assumes
#
# $$
# \begin{aligned}
# X_1, \ldots, X_N & \sim \mathcal{N}(\mu_X, \sigma^2) \\\\
# Y_1, \ldots, Y_M & \sim \mathcal{N}(\mu_Y, \sigma^2) \\\\
# X_1, \ldots, X_N, Y_1, \ldots, Y_M & \text { are mutually independent }
# \end{aligned}
# $$
#
# and tests the null hypothesis $\mu_X = \mu_Y$.
#
# *Remark*.
# In the above, it is assumed that the variance for both populations is the same.
# [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test) relaxes this assumption but is not covered here.
#
# The test statistic is
#
# $$
# T = \frac{\overline{X} - \overline{Y}}{s_p \cdot \sqrt{\frac{1}{N} + \frac{1}{M}}}
# $$
#
# where $\overline{X}$ and $\overline{Y}$ are the sample means of the two populations and
#
# $$
# s_p = \sqrt{ \frac{\left(N - 1\right) s_X^2 + \left(M - 1\right) s_Y^2}{N + M - 2} }
# $$
#
# is the pooled standard deviation.
# Similar to the Bessel corrected standard deviation of the previous section, $s_p^2$ is an unbiased estimate of $\sigma^2$.
# It can be shown that $T$ is a Student's t-distribution with $\nu = N + M - 2$ degrees of freedom.
# Similar to the previous section, the p-value corresponding to a test that rejects the null hypothesis when the magnitude of the test statistic is sufficiently large is
#
# $$
# \text{p-value} = 2F_{N + M - 2}(-|t|)
# $$
#
# where $t$ is the observed value of the test statistic.
#
# **Example (Running shoes, cont'd).**
# Continuing with our example, instead of having each runner run with *both*  SpeedBoot v1.0 and v2.0, ShoeScoop could have had each runner run with *either* SpeedBoot v1.0 or v2.0.
# The results of this alternate experiment along with the test statistic ($t$) and p-value are given below.

# %% tags=["no_input"]
np.random.seed(42)
N = 10
M = 13
X = np.round(
    np.random.normal(loc=60.0, scale=15.0, size=N)
    + np.random.normal(loc=0.0, scale=1.0, size=N),
    decimals=2,
)
Y = np.round(
    np.random.normal(loc=58.5, scale=(15.0**2 + 1.0**2) ** 0.5, size=M)
    + np.random.normal(loc=0.0, scale=1.0, size=M),
    decimals=2,
)

tabulate(
    itertools.zip_longest(np.arange(N), X, np.arange(M), Y),
    headers=(
        "SpeedBoot v1.0 Runner ($n$)",
        "Time ($X_n$)",
        "SpeedBoot v2.0 Runner ($m$)",
        "Time ($Y_m$)",
    ),
    tablefmt="html",
)


# %%
def indep_ttest(X: NDArray, Y: NDArray) -> tuple[float, float]:
    """Independent (a.k.a. two sample) t-test (should give results identical to `scipy.stats.ttest_ind`)."""
    (N,) = X.shape
    (M,) = Y.shape
    s_p = (
        ((N - 1) * X.std(ddof=1) ** 2 + (M - 1) * Y.std(ddof=1) ** 2) / (N + M - 2)
    ) ** 0.5
    t = (np.mean(X) - np.mean(Y)) / (s_p * (1.0 / N + 1.0 / M) ** 0.5)
    pvalue = 2.0 * scipy.stats.t.cdf(-np.abs(t), df=N + M - 2)
    return t.item(), pvalue.item()


t, pvalue = indep_ttest(X, Y)
print(f"{t=}, {pvalue=}")

# %% tags=["no_cell"]
expected = scipy.stats.ttest_ind(X, Y)
assert np.isclose(expected.statistic, t)
assert np.isclose(expected.pvalue, pvalue)


# %% [markdown]
# *Remark*.
# If we are in a situation where the assumptions of a paired t-test are satisfied, we should always opt for the paired t-test over the independent t-test as the latter is expected to produce a larger p-value (thereby making it harder to reject the null hypothesis).
# The reason for this is intuitive: in this scenario, applying the independent t-test is akin to discarding any information about which samples are paired.
#
# ## One sample t-test
#
# Given observations $X_1, \ldots, X_N$, a *one sample t-test* assumes
#
# $$
# X_1, \ldots, X_N \sim \mathcal{N}(\mu, \sigma^2) \text{ are i.i.d.}
# $$
#
# and tests the null hypothesis $\mu = \mu_0$ for some particular choice of $\mu_0$.
#
# This test reduces to the paired t-test with $Y_1 = \cdots = Y_M = \mu_0$ and as such, we omit the analysis here.


# %%
def onesamp_ttest(X: NDArray, mean: float) -> tuple[float, float]:
    """One sample t-test (should give results identical to `scipy.stats.ttest_1samp`)."""
    return dep_ttest(X, np.full_like(X, mean))


# %% tags=["no_cell"]
np.random.seed(42)
mean = 1.0
X = np.random.randn(20) + mean

t, pvalue = onesamp_ttest(X, mean)

expected = scipy.stats.ttest_1samp(X, mean)
assert np.isclose(expected.statistic, t)
assert np.isclose(expected.pvalue, pvalue)
