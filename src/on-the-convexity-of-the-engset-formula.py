# %% [raw]
# +++
# date = 2026-09-28
# title = "On the convexity of the Engset formula"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# ## Introduction and result
#
# The Engset formula gives a way of estimating how likely it is that a pool of servers will all be busy when traffic from one of finitely many sources arrives.
# The finitude of the number of sources of traffic is exactly what makes the Engset formula special; in the limit of the number of traffic sources, there are cheaper approximations.
#
# Questions like this one are the subject of the field of queueing theory.
# For an introduction, I recommend reading the Wikipedia pages on [queueing theory](https://en.wikipedia.org/wiki/Queueing_theory) and the [Engset formula](https://en.wikipedia.org/wiki/Engset_formula) and following the references therein.
#
# This short note focuses on a specific form of the Engset formula which is typically used when only the following quantities are known:
#
# * the (integer) number of servers $m \geq 1$;
# * the (integer) number of sources of traffic $N > m$;
# * the [offered traffic](https://en.wikipedia.org/wiki/Erlang_(unit)) per-source $\alpha > 0$.
#
# In this case, the Engset formula is an implicit equation of the form $P = f(P)$.
# The $P$ that solves this equation is the so-called "blocking probability" (i.e., the probability that all resources are busy).
# The function $f$ is given by
# $$
#   f(P) = \frac{
#       \binom{N - 1}{m}
#   }{
#       \sum_{j = 0}^m \binom{N - 1}{j} \left(
#           1/\alpha + P - 1
#       \right)^{m - j}
#   }.
# $$
#
# The form above appears in a note by Engset[^engset] and is reproduced in Driksna et al.[^driksna] along with numerical tables.
# It also appears in various other works[^rubas][^kubasik][^keshav].
#
# It turns out that the Engset formula is convex! Specifically...
#
# **Theorem.** _$f$ is convex on the nonnegative half-line $[0, \infty)$._
#
# ## Why care about convexity?
#
# Over a decade ago, [Tommy Carpenter](https://tommy.carpenternj.com) and I collaborated on studying the Engset formula above.
# Our project was born of an observation by Tommy: applying a fixed point iteration to $f$ often resulted in divergence.
#
# The resulting paper[^azimzadeh] characterized the instability of the fixed point iteration and suggested Newton's method as an alternative to compute the blocking probability.
# Specifically, the paper established that $f$ is convex on the nonnegative half-line whenever $\alpha \leq 1$.
# Convexity was in turn used to guarantee the convergence of Newton's method independent of the initial guess.
#
# Even though we didn't have a proof at the time of writing, we conjectured[^azimzadeh] based on numerical evidence that convexity (and hence convergence) should be independent of the choice of $\alpha > 0$.
# Now, we finally have an answer!
#
# ## Proof
#
# _Disclaimer_. Proof development was assisted by ChatGPT.
#
# Throughout, we use $(y_k)$ as shorthand for a sequence of numbers $y_0, y_1, \ldots$
# We call a nonnegative sequence $(y_k)$ _log-concave_ if $y_k^2 \geq y_{k - 1} y_{k + 1}$ at all interior indices $k$.
# We say $(y_k)$ has an _internal zero_ if there exist indices $i < j < k$ such that $y_j = 0$ and $y_i, y_k \neq 0$.
#
# The following specializes a more general result by Keilson[^keilson].
# It is used to prove the subsequent lemma.
#
# **Proposition.**
# _Let $K$ be a nonnegative integer random variable with probability mass function $k \mapsto p_k$._
# _If the sequence $(p_k)$ is log-concave with no internal zeros, then $2 (\mathbb{E} K)^2 \geq \mathbb{E} \left[ K \left(K - 1 \right) \right]$._
#
# **Lemma.**
# _Let $m$ be a positive integer, $(a_k)$ be a positive and log-concave sequence, and_
# $$
#   Q(x) = \sum_{k = 0}^m a_k x^k.
# $$
# _Then, $1 / Q$ is convex on the nonnegative half-line._
#
# _Proof._
# Fix a positive number $x$ and let $K_x$ be the nonnegative integer random variable with PMF
# $$
#   \mathbb{P}(K_x = k) = \frac{a_k x^k}{Q(x)}.
# $$
# By direct computation,
# $$
#   \mathbb{E} K_x = \frac{x Q^\prime(x)}{Q(x)}
#   \text{ and }
#   \mathbb{E} \left[ K_x \left( K_x - 1 \right) \right] = \frac{x^2 Q^{\prime \prime}(x)}{Q(x)}.
# $$
# Since $(a_k)$ is log-concave, so too is the PMF of $K_x$.
# Applying Keilson's result at each positive $x$,
# $$
#   2 (Q^\prime)^2 - Q Q^{\prime \prime} \geq 0 \text{ on } (0, \infty).
# $$
# Therefore,
# $$
#   \left( \frac{1}{Q} \right)^{\prime \prime} = \frac{2 \left(Q^\prime\right)^2 - Q Q^{\prime \prime}}{Q^3} \geq 0 \text{ on } (0, \infty).
# $$
# Since $Q(0) = a_0 > 0$, convexity extends to zero by continuity.
# $\blacksquare$
#
# We are now ready to prove the main result.
#
# _Proof (Engset formula convexity)_.
# If $m = N - 1$, then $f(P) = (1/\alpha + P)^{-m}$ and the desired result is trivial.
# Therefore, proceed assuming $m < N - 1$.
# Let
# $$
#   Q(x)
#   =\sum_{j=0}^{m}\binom{N-1}{j}(x-1)^{m-j}.
# $$
# It follows that, by power series manipulations,
# $$
#   Q(x)
#   = [z^m] \frac{(1 + z)^{N - 1}}{1 - \left(x - 1\right) z}
#   = [z^m] \sum_{k \geq 0} x^k z^k \left(1 + z\right)^{N - k - 2}
#   = \sum_{k = 0}^m \binom{N - k - 2}{m - k} x^k.
# $$
# Next, note that $f(P)$ can be written as a constant times the reciprocal of $Q$ evaluated at $x = 1 / \alpha + P$.
# Since $Q$ satisfies the requirements of the previous lemma, the desired result follows.
# $\blacksquare$
#
# ## References
#
# [^engset]: T. Engset, "Die Wahrscheinlichkeitsrechnung zur Bestimmung der Wähleranzahl in automatischen Fernsprechämtern," _Elektrotechnische Zeitschrift_ 39, no. 31 (1918): 304-306.
# [^driksna]: V. V. Driksna and E. G. Wormald, ["Engset Traffic Tables,"](http://www.coxhill.com/trlhistory/media/Telecommmunication%20Journal%20of%20Australia/The%20Telecommunication%20Journal%20of%20Australia%20Vol%2016%20No%202%20JUNE%201966.pdf) _The Telecommunication Journal of Australia_ 16, no. 2 (June 1966): 154-156.
# [^rubas]: J. Rubas, ["Dimensioning of Alternative Routing Networks Offered Smooth Traffic,"](https://www.coxhill.com/trlhistory/media/Australian%20Telecommunication%20Research/Australian%20Telecommunication%20Research%20Vol%2011%20No%201%201977.pdf) _Australian Telecommunications Research_ 11, no. 1 (1977): 38-44.
# [^kubasik]: J. Kubasik, "On Some Numerical Methods for the Computation of Erlang and Engset Functions," in _Proceedings of the Eleventh International Teletraffic Congress_ (ITC-11), Kyoto, Japan (1985), 958-964.
# [^keshav]: S. Keshav, _An Engineering Approach to Computer Networking: ATM Networks, the Internet, and the Telephone Network_ (Reading, MA: Addison-Wesley, 1997).
# [^azimzadeh]: P. Azimzadeh and T. Carpenter, ["Fast Engset Computation,"](https://arxiv.org/pdf/1511.00291) _Operations Research Letters_ 44, no. 3 (2016): 313-318.
# [^keilson]: J. Keilson, ["A threshold for log-concavity for probability generating functions and associated moment inequalities,"](https://doi.org/10.1214/aoms/1177692406) _The Annals of Mathematical Statistics_ (1972): 1702-1708.
