# %% [raw]
# +++
# aliases = [
#   "/blog/2023/a_pedagogical_introduction_to_the_determinant"
# ]
# date = 2023-12-16
# title = "A pedagogical introduction to the determinant"
# +++

# %% tags=["no_cell"]
import numpy as np

from _boilerplate import init

init()


# %% [markdown]
# ## Motivation
#
# The determinant of a matrix is typically introduced in an undergraduate linear algebra course via either the Leibniz Formula or a recurrence relation arising from the Leibniz Formula.
# Pedagogically, it is better to introduce the determinant as a mapping which satisfies some desirable properties and only *then* show that it is equivalent to the Leibniz Formula.
# This short expository post attempts to do just that.
#
# ## Determinant
#
# A *determinant function* is a mapping $\det$ from the square complex matrices to complex numbers satisfying the following properties:
#
# 1. If the matrix $A^\prime$ is obtained by swapping two rows of $A$, then $\det A^\prime = - \det A$.
# 2. If the matrix $A^\prime$ is obtained by multiplying a single row of $A$ by a constant $c$, then $\det A^\prime = c \det A$.
# 3. If the matrix $A^\prime$ is obtained by adding a multiple of a row of $A$ to another (not the same) row, then $\det A^\prime = \det A$.
# 4. $\det I = 1$.
#
# The first three points above correspond to the three [elementary row operations](https://en.wikipedia.org/wiki/Elementary_matrix).
#
# **Proposition.**
# Let $\det$ be a determinant function and $A$ be a square complex matrix whose rows are linearly dependent.
# Then, $\det A = 0$.
#
# *Proof*.
# In this case, we can perform a sequence of elementary row operations (excluding multiplying a row by $c = 0$) that result in a row consisting of only zeros.
# The result then follows by property (2).
#
# Indeed, by performing elimination to reduce the matrix into either the identity or a matrix with at least one row of zeros, we can unambiguously define a determinant function (note that we have not yet proven that such a function is unique).
# The code below does just that, proving the existence of a determinant function.
# For now, we refer to this as the *canonical determinant*.
#
# *Remark*.
# The code below operates on floating point numbers.
# The definition of the *canonical determinant* should be understood to be the "algebraic" version of this code that runs without deference to floating point error.


# %%
def det(mat: np.ndarray) -> float:
    """Computes a determinant.

    This algorithm works by eliminating the strict lower triangular part of the
    matrix and then eliminating the strict upper triangular part of the matrix.
    This elimination is done using row operations, while keeping track of any
    swaps that may change the sign parity of the determinant.

    If you are already familiar with the determinant, you will note that
    eliminating the strict upper triangular part is not necessary. Even if this
    algorithm was optimized to remove that step, this is still not a performant
    way to compute determinants!

    Parameters
    ----------
    mat
        A matrix

    Returns
    -------
    Determinant
    """
    m, n = mat.shape
    assert m == n

    mat = mat.copy()

    sign = 1
    for _ in range(2):
        for j in range(n):
            # Find pivot element
            p = -1
            for i in range(j, n):
                if not np.isclose(mat[i, j], 0.0):
                    p = i
                    break
            if p < 0:
                continue

            # Swap
            if j != p:
                r = mat[p].copy()
                mat[p] = mat[j]
                mat[j] = r
                sign *= -1

            # Eliminate
            for i in range(j + 1, n):
                if not np.isclose(mat[i, j], 0.0):
                    mat[i] -= mat[p] * mat[i, j] / mat[p, j]

        mat = mat.T

    return float(sign) * np.diag(mat).prod().item()


# %% tags=["no_cell"]
np.random.seed(0)
mat = np.random.randn(64, 64)
np.testing.assert_allclose(det(mat), np.linalg.det(mat))

# %% [markdown]
# ## Alternating multilinear maps
#
# *Notation*.
# For a set $\mathcal{A}$, we write $A \equiv (a_1, \ldots, a_n)$ to denote an element of $\mathcal{A}^n$.
#
# **Definition (Alternating multilinear map).**
# Let $\mathcal{A}$ and $\mathcal{B}$ be vector spaces.
# An *alternating multilinear map* is a [multilinear map](https://en.wikipedia.org/wiki/Multilinear_map) $f: \mathcal{A}^n \rightarrow \mathcal{B}$ that satisfies $f(A) = 0$ whenever $a_i = a_{i + 1}$ for some $i < n$.
#
# *Notation*.
# Let $\sigma$ be a [permutation](https://en.wikipedia.org/wiki/Permutation) of {1, ..., n}.
# Since $A$ in $\mathcal{A}^n$ can be thought of as a function from {1, ..., n} to $\mathcal{A}$, we write $A \circ \sigma \equiv (a_{\sigma(1)}, \ldots, a_{\sigma(n)})$ to denote a permutation of the elements of $A$.
#
# **Proposition (Transposition parity).**
# Let $f$ be an alternating multilinear map.
# Let $\sigma$ be a [transposition](https://en.wikipedia.org/wiki/Cyclic_permutation#Transpositions) (a permutation which swaps two elements).
# Then, $f(A) = -f(A \circ \sigma)$.
#
# *Proof*.
# Let $i < j$ denote the swapped indices in the transposition.
# Fix $A$ and let
#
# $$g(x, y) \equiv f(a_1, \ldots, a_{i - 1}, x, a_{i + 1}, \ldots, a_{j - 1}, y, a_{j + 1}, \ldots, a_n).$$
#
# It follows that
#
# $$
# \begin{aligned}
# g(x, y) + g(y, x)
# & = g(x, y) + g(y, y) + g(y, x) + g(x, x) \\\\
# & = g(x + y, y) + g(x + y, x) \\\\
# & = g(x + y, x + y) \\\\
# & = 0
# \end{aligned}
# $$
#
# and hence $g(x, y) = -g(y, x)$, as desired. $\blacksquare$
#
# **Corollary.**
# Let $f$ be an alternating multilinear map.
# Then $f(A) = 0$ whenever $a_i = a_j$ for some $(i, j)$ with $i < j$.
#
# *Proof*.
# Let $\sigma$ be the transposition which swaps indices $i + 1$ and $j$.
# Then, $f(A) = -f(A \circ \sigma) = 0$. $\blacksquare$
#
# **Corollary.**
# Let $f$ be an alternating multilinear map and $\sigma$ be a permutation.
# Then, $f(A) = \operatorname{sgn}(\sigma) f(A \circ \sigma)$ where $\operatorname{sgn}(\sigma)$ is the [parity of the permutation](https://en.wikipedia.org/wiki/Parity_of_a_permutation).
#
# *Proof*.
# The result follows from the fact that a permutation can be written as a composition of transpositions. $\blacksquare$
#
# **Proposition.**
# A multilinear map $f:\mathcal{A}^{n}\rightarrow\mathcal{B}$ is alternating multilinear if and only if $f(A)=0$ whenever $a_{1},\ldots,a_{n}$ are linearly dependent.
#
# *Proof*.
# Suppose the map is alternating multilinear. Let $a_{1},\ldots,a_{n}$ be linearly dependent so that, without loss of generality, $a_{1}=\sum_{i>1}\alpha_{i}a_{i}$.
# By linearity,
#
# $$
# f(A)=\sum_{i>1}\alpha_{i}f(a_{i},a_{2},\ldots,a_{n})=0.
# $$
#
# The converse is trivial. $\blacksquare$
#
# ## The Leibniz formula
#
# *Notation*.
# If $\mathcal{A} = \mathbb{C}^n$, then $\mathcal{A}^n$ is isomorphic to the set of $n \times n$ complex matrices.
# In light of this, an element in $\mathcal{A}^n$ can be considered as a matrix $A \equiv (a_{ij})$ or as a tuple $A \equiv (a_1, \ldots, a_n)$ consisting of the rows of said matrix.
#
# **Proposition (Uniqueness).**
# Let $f: (\mathbb{C}^n)^n \rightarrow \mathbb{C}$ be an alternating multilinear map such that $f(I) = 1$.
# Then,
#
# $$
# f(A) = \sum_{\sigma \in S_n} \operatorname{sgn}(\sigma) a_{1 \sigma(1)} \cdots a_{n \sigma(n)}.\tag{Leibniz Formula}
# $$
#
# where $S_n$ is the set of all permutations on {1, ..., n}.
#
# *Proof*.
# First, note that
#
# $$
# f(A) = f\biggl(\sum_j a_{1j} e_j, \ldots, \sum_j a_{nj} e_j\biggr) \\
#      = \sum_{1 \leq j_1,\ldots,j_n \leq n} a_{1 j_1} \cdots a_{n j_n} f(e_{j_1}, \ldots, e_{j_n}).
# $$
#
# Since $f$ is alternating multilinear and hence equal to zero whenever any of its two inputs are equal, we can restrict our attention to the permutations:
#
# $$
# f(A) = \sum_{\sigma \in S_n} a_{1 \sigma(1)} \cdots a_{n \sigma(n)} f(e_{\sigma(1)}, \ldots, e_{\sigma(n)}).
# $$
#
# Since $f$ is alternating multilinear, we can change the order of its inputs so long as we count the number of transpositions and use that to account for a possible sign-change:
#
# $$
# f(A) = \sum_{\sigma \in S_n} \operatorname{sgn}(\sigma) a_{1 \sigma(1)} \cdots a_{n \sigma(n)} f(I).
# $$
#
# Using the assumption $f(I) = 1$, the desired result follows. $\blacksquare$
#
# *Remark*.
# $\operatorname{sgn}(\sigma)$ is sometimes represented as $\epsilon_{i_1 \ldots i_n}$ where $i_j = \sigma(j)$.
# This is called the [Levi-Civita symbol](https://en.wikipedia.org/wiki/Levi-Civita_symbol).
# Using this symbol and [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation), the Leibniz Formula becomes
#
# $$
# \epsilon_{i_1 \ldots i_n} a_{1 i_1} \cdots a_{n i_n}.
# $$
#
# **Proposition.**
# A determinant function is multilinear.
#
# *Proof.*
# Let $A$ be a square complex matrix and $h$ be a vector.
# It is sufficient to show that
#
# $$
# \det A+\det(h,a_{2},\ldots,a_{n})=\det(a_{1}+h,a_{2},\ldots,a_{n}).
# $$
#
# Suppose the rows of $A$ are linearly dependent.
# Without loss of generality,
# write $a_{1}=\sum_{i>1}\alpha_{i}a_{i}$ and $h=b+\sum_{i>1}\beta_{i}a_{i}$
# where $b$ is orthogonal to the $a_{i}$.
# Then, $\det A=0$.
# Moreover,
#
# $$
# \det(h,a_{2},\ldots,a_{n})=\det\biggl(b+\sum_{i>1}\beta_{i}a_{i},a_{2},\ldots,a_{n}\biggr)=\det(b,a_{2},\ldots,a_{n})
# $$
#
# and
#
# $$
# \det(a_{1}+h,a_{2},\ldots,a_{n})=\det\biggl(b+\sum_{i>1}\left(\alpha_{i}+\beta_{i}\right)a_{i},a_{2},\ldots,a_{n}\biggr)=\det(b,a_{2},\ldots,a_{n}),
# $$
#
# as desired.
#
# Suppose the rows of $A$ are linearly independent.
# It follows that we can write $h=\sum_{i}\beta_{i}a_{i}$.
# Then,
#
# $$
# \begin{aligned}
# \det A+\det(h,a_{2},\ldots,a_{n}) & =\det A+\det\biggl(\sum_{i}\beta_{i}a_{i},a_{2},\ldots,a_{n}\biggr)\\\\
#  & =\det A+\det\biggl(\beta_{1}a_{1},a_{2},\ldots,a_{n}\biggr)\\\\
#  & =\det\biggl(\left(1+\beta_{1}\right)a_{1},a_{2},\ldots,a_{n}\biggr)\\\\
#  & =\det\biggl(a_{1}+\sum_{i}\beta_{i}a_{i},a_{2},\ldots,a_{n}\biggr)\\\\
#  & =\det\biggl(a_{1}+h,a_{2},\ldots,a_{n}\biggr). \blacksquare
# \end{aligned}
# $$
#
# **Corollary.**
# A determinant function is an alternating multilinear map.
#
# **Corollary.**
# There is only one determinant function and it is given by the Leibniz Formula.
#
# ## Determinant properties
#
# We can now use the Leibniz Formula to derive various properties of the determinant.
# The following results are concerned with complex matrices $A \equiv (a_{ij})$ and $B \equiv (b_{ij})$.
#
# **Proposition.**
# $\det A = \det A^\intercal$.
#
# *Proof*.
#
# $$
# \det A
# =\sum_{\sigma}\operatorname{sgn}(\sigma)\prod_{i}a_{i\sigma(i)}
# =\sum_{\sigma}\operatorname{sgn}(\sigma)\prod_{i}a_{\sigma^{-1}(i)\sigma(\sigma^{-1}(i))}
# =\sum_{\sigma}\operatorname{sgn}(\sigma^{-1})\prod_{i}a_{\sigma^{-1}(i)i}
# =\det A^{\intercal}. \blacksquare
# $$
#
# *Notation*.
# For a matrix $A$, let $A^{(i, j)}$ be the same matrix after the simultaneous removal of its $i$-th row and $j$-th column.
#
# **Lemma.**
#
# $$
# \det A = \sum_j \left( -1 \right)^{j - 1} a_{1j} \det A^{(1, j)}
# $$
#
# *Proof*.
# We demonstrate the idea for a $3\times3$ matrix; the generalization is straight-forward.
#
# Using multilinearity,
#
# $$
# \begin{aligned}
# \det\begin{pmatrix}a_{11} & a_{12} & a_{13}\\\\
# a_{21} & a_{22} & a_{23}\\\\
# a_{31} & a_{32} & a_{33}
# \end{pmatrix} & =a_{11}\det\begin{pmatrix}1 & 0 & 0\\\\
# a_{21} & a_{22} & a_{23}\\\\
# a_{31} & a_{32} & a_{33}
# \end{pmatrix}+a_{12}\det\begin{pmatrix}0 & 1 & 0\\\\
# a_{21} & a_{22} & a_{23}\\\\
# a_{31} & a_{32} & a_{33}
# \end{pmatrix}+a_{13}\det\begin{pmatrix}0 & 0 & 1\\\\
# a_{21} & a_{22} & a_{23}\\\\
# a_{31} & a_{32} & a_{33}
# \end{pmatrix}\\\\
#  & =a_{11}\det\begin{pmatrix}1 & 0 & 0\\\\
# a_{21} & a_{22} & a_{23}\\\\
# a_{31} & a_{32} & a_{33}
# \end{pmatrix}-a_{12}\det\begin{pmatrix}1 & 0 & 0\\\\
# a_{22} & a_{21} & a_{23}\\\\
# a_{32} & a_{31} & a_{33}
# \end{pmatrix}+a_{13}\det\begin{pmatrix}1 & 0 & 0\\\\
# a_{23} & a_{21} & a_{22}\\\\
# a_{33} & a_{31} & a_{32}
# \end{pmatrix}
# \end{aligned}
# $$
#
# Moreover, by the Leibniz Formula,
#
# $$
# \begin{aligned}
# \det\begin{pmatrix}1 & 0 & 0\\
# a_{21} & a_{22} & a_{23}\\
# a_{31} & a_{32} & a_{33}
# \end{pmatrix}
# &=\sum_{\sigma}\operatorname{sgn}(\sigma)a_{1\sigma(1)}a_{2\sigma(2)}a_{3\sigma(3)}\\\\
# &=\sum_{\sigma\colon\sigma(1)=1}\operatorname{sgn}(\sigma)a_{2\sigma(2)}a_{3\sigma(3)}\\\\
# &=\det\begin{pmatrix}a_{22} & a_{23}\\
# a_{32} & a_{33}
# \end{pmatrix}.
# \end{aligned}
# $$
#
# The remaining terms are handled similarly. $\blacksquare$
#
# **Proposition (Cofactor expansion).**
# For any $i$ between $1$ and $n$ (inclusive),
#
# $$
# \det A = \sum_j \left( -1 \right)^{i + j} a_{ij} \det A^{(i, j)}.
# $$
#
# *Proof*.
# Recalling that the determinant flips signs when any two rows are swapped, we can perform a sequence of $i - 1$ transpositions to move $a_i$, the $i$-th row of the matrix, to the "top" and apply the previous lemma:
#
# $$
# \left(-1\right)^{i-1}\det A=\det\begin{pmatrix}a_{i}^{\intercal}\\
# a_{1}^{\intercal}\\
# a_{2}^{\intercal}\\
# \vdots\\
# a_{i-1}^{\intercal}\\
# a_{i+1}^{\intercal}\\
# \vdots\\
# a_{n}^{\intercal}
# \end{pmatrix}. \blacksquare
# $$
#
# **Corollary.**
# If $A$ is either lower or upper triangular, $\det A = \prod_i a_{ii}$.
#
# *Proof*.
# First, note that it is sufficient to consider the lower triangular case since the transpose of an upper triangular matrix is lower triangular.
# The result then follows from performing a cofactor expansion along the first row inductively. $\blacksquare$
#
# **Proposition.**
#
# $$\det(AB) = \det A \det B$$
#
# *Proof*.
# If either $A$ or $B$ are singular, the claim is trivial since both sides are zero.
# Therefore, proceed assuming $A$ and $B$ are nonsingular.
#
# As with the construction of the canonical determinant, we can write
#
# $$
# I=E_{k}\cdots E_{1}A
# $$
#
# where $E_{1},\ldots,E_{k}$ are a sequence of elementary row operations.
# It is easy to see that elementary row operations are nonsingular and their inverses are themselves elementary row operations.
# Therefore, $A$ can be written as a product of elementary row operations.
# To arrive at the desired result, it is sufficient to show that for any sequence of row operations $E_{1}^{\prime},\ldots,E_{k}^{\prime}$
# there exists a constant $\alpha$ such that for any matrix $M$
#
# $$
# \det(E_{1}^{\prime}\cdots E_{k}^{\prime}M)=\alpha\det M. \blacksquare
# $$
#
# **Corollary.**
# The determinant of an $n \times n$ complex matrix is the product of its $n$ (possibly non-unique) eigenvalues.
#
# *Proof*.
# Let $A$ be an $n \times n$ complex matrix and denote by $A = P^{-1} J P$ its [Jordan normal form](https://en.wikipedia.org/wiki/Jordan_normal_form).
# Since the matrix $J$ has the eigenvalues $\lambda_1, \ldots, \lambda_n$ of $A$ on its diagonal and is upper triangular,
#
# $$\det A = \det P^{-1} \det J \det P = \det J = \prod_{i = 1}^n \lambda_i. \blacksquare$$
