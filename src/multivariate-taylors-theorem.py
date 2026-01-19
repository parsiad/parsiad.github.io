# %% [raw]
# +++
# aliases = [
#   "/blog/2021/multivariate_taylors_theorem"
# ]
# date = 2021-07-20
# title = "Multivariate Taylor's Theorem"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# ## Motivation
#
# This short post derives Taylor's Theorem for multivariate functions by employing Taylor's Theorem in a single variable.
#
# ## Derivation
#
# Let $f : \mathbb{R}^d \rightarrow \mathbb{R}$.
# For vectors $x$ and $v$ in $\mathbb{R}^d$, define $g : \mathbb{R} \rightarrow \mathbb{R}$ by $g(t) = f(x + tv)$.
# If $g$ is $K$ times differentiable at zero, Taylor's theorem in 1d tells us
#
# $$
# \tag{1}
#     f(x + tv)
#     = g(t)
#     = \sum_{k = 0}^K \frac{t^k}{k!} g^{(k)}(0)
#     + o(t^K) \text{ as } t \rightarrow 0.
# $$
#
# Suppose
#
# $$
# \tag{2}
#     g^{(k)}(t)
#     = \sum_{i_1, \ldots, i_k}
#       v_{i_1} \cdots v_{i_k}
#       \frac{\partial^k f}{\partial x_{i_1} \cdots x_{i_k}}(x + tv).
# $$
#
# By chain rule,
#
# $$
#     g^{(k + 1)}(t)
#     = \sum_{i_1, \ldots, i_k}
#       v_{i_1} \cdots v_{i_k}
#       \left \langle
#           v,
#           \nabla \left[
#               \frac{\partial^k f}{\partial x_{i_1} \cdots x_{i_k}}
#           \right] (x + tv)
#       \right \rangle.
# $$
#
# Simplifying, we arrive at (2) with $k$ replaced by $k + 1$.
# Since (2) is trivially satisfied at $k = 1$, it follows by induction that (2) holds for all positive integers $k$.
#
# The form (2) is redundant since, assuming the conditions of [Clairaut's theorem](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives#Sufficiency_of_twice-differentiability), partial derivatives commute (e.g., $f_{x_1 x_2} = f_{x_2 x_1}$).
# For a multi-index $\alpha = (\alpha_1, \ldots, \alpha_d)$ in $\mathbb{Z}^d_{\geq 0}$, define $|\alpha| = \alpha_1 + \cdots + \alpha_d$ and
#
# $$
#     D^\alpha f
#     = \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1} \cdots \partial x_d^{\alpha_d}}.
# $$
#
# With this notation we can write (2) as
#
# $$
#     g^{(k)}(t)
#     = \sum_{|\alpha| = k}
#       \frac{k!}{\alpha_1! \cdots \alpha_d!}
#       v_1^{\alpha_1} \cdots v_d^{\alpha_d}
#       D^\alpha f(x + tv).
# $$
#
# Substituting this into (1), we obtain the desired Taylor polynomial:
#
# $$
#     f(x + tv)
#     = \sum_{k = 0}^K t^k
#       \sum_{|\alpha| = k}
#       \frac{1}{\alpha_1! \cdots \alpha_d!}
#       v_1^{\alpha_1} \cdots v_d^{\alpha_d}
#       D^\alpha f(x)
#       + o(t^K) \text{ as } t \rightarrow 0
# $$
#
# ## Remainder
#
# If, in addition, $g$ is $K + 1$ times differentiable, we can extend the Cauchy or Lagrange form of the remainder term to the multivariate setting.
# For example, the Lagrange form is
#
# $$
#     o(t^K) = \frac{t^{K + 1}}{\left( K + 1 \right)!} g^{(K + 1)}(\theta)
# $$
#
# where $\theta$ is some number between zero and $t$.
# Substituting (2) into the above, we can obtain by triangle inequality, the (loose) bound
#
# $$
#     o(t^K)
#     \leq \frac{
#              \left( d \left| t \right| \left \Vert v \right \Vert_\infty \right)^{K + 1}
#          }{\left( K + 1 \right)!}
#          \max_{|\alpha| = K + 1}
#          \left |D^\alpha f(x + \theta v) \right|.
# $$
