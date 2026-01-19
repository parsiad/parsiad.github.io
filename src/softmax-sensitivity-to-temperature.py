# %% [raw]
# +++
# aliases = [
#   "/blog/2022/softmax_sensitivity_to_temperature"
# ]
# date = 2022-11-14
# title = "Softmax sensitivity to temperature"
# +++

# %% [markdown]
# ## Review
#
# The [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is a way to transform a vector of real numbers $r$ into a probability vector $\sigma(r)$ while preserving the monotonicity of the coordinates:
#
# $$
# \sigma(r)_i \equiv \frac{\exp(r_i)}{\sum_j \exp(r_j)}.
# $$

# %% tags=["no_cell"]
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from _boilerplate import display_fig, init

init()


# %%
def softmax(r: ArrayLike) -> NDArray:
    """Computes the softmax of the input.

    If the input is a tensor, the sum is reduced along the first axis.
    """
    exp_r = np.exp(r)
    return exp_r / exp_r.sum(axis=0)


# %% [markdown]
# An application of the softmax is in training [neural](https://en.wikipedia.org/wiki/Neural_network) [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification) with classes $i = 1, 2, \ldots, k$.
# In this case, we have a neural network $x \mapsto f(x)$ which transforms a feature vector $x$ into a real-valued vector $f(x)$ with one coordinate for each of the $k$ classes.
# The predicted class is the coordinate of $f(x)$ with the maximum value.
# Training such a network is typically achieved by minimizing the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy):
#
# $$
# \sum_{(x, i)} -\log \sigma(f(x))_i
# $$
#
# where the sum is taken over feature-class pairs $(x, i)$ in the training set.
#
# ## Temperature
#
# Consider the softmax of the vector $r = (1, 2, 3)$:

# %%
r = np.array([1.0, 2.0, 3.0])
softmax(r)

# %% [markdown]
# We see that a majority of the probability mass is assigned to the last coordinate.
# What if this is undesirable?
# Let's examine what happens if we *attenuate* the vector before applying the softmax...

# %%
softmax(r / 10.0)

# %% [markdown]
# Note, that while the monotonicity is preserved, the probabilities tend to a uniform distribution.
# Conversely, if we *amplify* $r$ before applying the softmax, the probabilities tend towards an atom which places all of the mass on the largest coordinate:

# %%
softmax(r * 10.0)

# %% [markdown]
# In general, we define
#
# $$
# \sigma_T(r) \equiv \sigma(r / T)
# $$
#
# as the *softmax with temperature $T$*.
# Here, $T$ is a scalar that, just as above, controls the clustering of probabilities.
#
# This behavior is more clearly demonstrated in the plot below, where the softmax is computed for many values of the temperature:


# %% tags=["no_cell"]
def plot_softmax_vs_temp(x: ArrayLike, T_min: float = 0.1, T_max: float = 1_000.0):
    x = np.array(x)
    T = np.logspace(np.log10(T_min), np.log10(T_max))
    s = softmax(x[:, np.newaxis] / T[np.newaxis])
    x_str = ", ".join(f"{xi:g}" for xi in x)

    fig, ax = plt.subplots(constrained_layout=True)
    for i, (si, xi) in enumerate(zip(s, x)):
        label = rf"$\sigma_T(r)_{i + 1}$"
        opts = ":k" if np.any(x[:i] == xi) else "-"
        ax.semilogx(T, si, opts, label=label)

    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Probability")
    ax.set_title(f"Softmax of $r=({x_str})$ vs. Temperature")
    ax.legend()
    display_fig(fig)


# %% tags=["no_input"]
plot_softmax_vs_temp([1.0, 2.0, 3.0])

# %% [markdown]
# In the plot above, we can see the limiting behavior as $T \rightarrow 0$ is to place all of the mass on the atom corresponding to the largest coordinate of $r$.
# However, when there is no single largest coordinate, the limiting behavior is to evenly split the probability mass among the largest coordinates:

# %% tags=["no_input"]
plot_softmax_vs_temp([1.0, 2.0, 3.0, 3.0])

# %% [markdown]
# We can state our findings above rigorously.
# For brevity, define $\sigma_0(r) \equiv \lim_{T \rightarrow 0} \sigma_T(r)$ and $\sigma_\infty$ similarly.
# Let $e = (1, \ldots, 1)^\intercal$ be the ones vector.

# %% [markdown]
# **Proposition.**
# Let $r$ be a vector in $\mathbb{R}^k$.
# Let $m_i$ be equal to one if $r_i = \max_j r_j$ and zero otherwise.
# Then,
#
# $$
# \sigma_0(r) = \frac{m}{e^\intercal m}
# $$
#
# *Proof*.
# Let $\overline{r} \equiv \max_j r_j$.
# Fix $i$.
# If $r_i < \overline{r}$, then
#
# $$
# \sigma_{T}(r)_i
# \leq \frac{\exp(r_i / T)}{\exp(\overline{r} / T)}
# = \exp((r_i - \overline{r}) / T)
# $$
#
# which tends to zero as $T$ vanishes.
# Therefore, the only nonzero coordinates $i$ of $\sigma_0(r)$ are those satisfying $r_i = \overline{r}$.
# Since all such coordinates must be equal, the desired result follows from the fact that $\sigma_0(r)$ is a probability vector.
#
# **Proposition.**
# Let $r$ be a vector in $\mathbb{R}^k$.
# Then,
#
# $$
# \sigma_\infty(r) = \frac{e}{k}.
# $$
#
# *Proof*.
# This follows immediately from $\lim_{T \rightarrow \infty} \exp(r_i / T) = 1$.
