# %% [raw]
# +++
# aliases = [
#   "/blog/2023/motivating_the_cross_entropy_loss"
# ]
# date = 2023-09-23
# title = "Motivating the cross-entropy loss"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# ## Introduction
#
# In machine learning, the cross-entropy loss is frequently introduced without explicitly emphasizing its underlying connection to the likelihood of a categorical distribution.
# Understanding this link can greatly enhance one's grasp of the loss and is the topic of this short post.
#
# ## Prerequisites
#
# * [maximum likelihood estimator (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
#
# ## Categorical distribution likelihood
#
# Consider an experiment in which we roll a (not necessarily fair) $K$-sided die.
# The result of this roll is an integer between $1$ and $K$ (inclusive) corresponding to the faces of the die. Let $q(k)$ be the probability of seeing the $k$-th face.
# What we have described here, in general, is a categorical random variable: a random variable which takes one of a finite number of values.
# Repeating this experiment multiple times yields IID random variables $X_{1},\ldots,X_{N}\sim\operatorname{Categorical}(q)$.
#
# Performing this experiment a finite number of times $N$ does not allow us to introspect $q$ precisely, but it does allow us to estimate it.
# One way to approximate $q(k)$ is by counting the number of times the die face $k$ was observed and normalizing the result:
#
# $$
# p(k)=\frac{1}{N}\sum_{n}[X_{n}=k]\tag{empirical PMF}
# $$
#
# where $[\cdot]$ is the [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket). Since $Y_{n}=[X_{n}=k]$ is itself a random variable (an indicator random variable), the law of large numbers tells us that $p(k)$ converges (a.s.) to $\mathbb{E}Y_{1}=\mathbb{P}(X_{n}=k)=q(k)$.
#
# The likelihood of $q$ is
#
# $$
# \mathcal{L}(q)=\prod_{n}\prod_{k}q(k)^{[X_{n}=k]}=\prod_{k}q(k)^{\sum_{n}[X_{n}=k]}=\prod_{k}q(k)^{Np(k)}
# $$
#
# and hence its log-likelihood is
#
# $$
# \ell(q)=\log\mathcal{L}(q)=\sum_{k}Np(k)\log q(k)\propto\sum_{k}p(k)\log q(k).
# $$
#
# **Proposition**. The MLE for the parameter of the categorical distribution is the empirical PMF above.
#
# *Proof*. Consider the program
#
# $$
# \begin{aligned}
# \min_{q} & -\ell(q)\\\\
# \text{subject to} & \sum_{k}q(k)-1=0.
# \end{aligned}
# $$
#
# The [Karush--Kuhn--Tucker stationarity condition](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) is
#
# $$
# -\frac{p(k)}{q(k)}+\lambda=0\text{ for }k=1,\ldots,K.
# $$
#
# In other words, the MLE $\hat{q}$ is a multiple of $p$.
# Since the MLE needs to be a probability vector, $\hat{q} = p$.
#
# ## Cross-entropy
#
# The cross-entropy between $q$ relative to $p$ is
#
# $$
# H(p, q) = - \mathbb{E}_{X \sim p} [ \log q(X) ].
# $$
#
# The choice of logarithm base yields different units:
# * base 2: [bits](https://en.wikipedia.org/wiki/Bit)
# * base e: [nats](https://en.wikipedia.org/wiki/Nat_(unit))
# * base 10: [hartleys](https://en.wikipedia.org/wiki/Hartley_(unit))
#
# When $p$ and $q$ are probability mass functions (PMFs), the cross-entropy reduces to
#
# $$
# H(p, q) = - \sum_x p(x) \log q(x)
# $$
#
# which is exactly the (negation of the) log-likelihood we encountered above.
# As such, one can intuit that minimizing $q$ in the cross-entropy yields a distribution that is similar to $p$.
# In other words, **the cross-entropy is an asymmetric measure of dissimilarity between $q$ and $p$.**
#
# The [Kullback--Leibler (KL) divergence](https://en.m.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is another such measure:
#
# $$
# D_{\mathrm{KL}}(p\Vert q)
# =\mathbb{E}_{p}\left[\log\frac{p(X)}{q(X)}\right]
# =H(p,q) - H(p,p).
# $$
#
# Minimizing the KL divergence is the same as minimizing the cross-entropy, but the KL divergence satisfies some nice properties that one would expect of a measure of dissimilarity.
# In particular,
# 1. $D_{\mathrm{KL}}(p\Vert q) \geq 0$
# 2. $D_{\mathrm{KL}}(p\Vert p) = 0$
#
# We proved the first inequality for PMFs by showing that the choice of $q = p$ maximizes the cross-entropy.
# The second inequality is trivial.
#
# ## Cross-entropy loss
#
# Statistical classification is the problem of mapping each input datum $x \in \mathcal{X}$ to a class label $y = 1, \ldots, K$.
# For example, in the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) classification task, each $x$ is a 32x32 color image and each $K = 10$ corresponding to ten distinct classes (e.g., airplanes, cats, trucks).
#
# A common parametric estimator for image classification tasks such as CIFAR-10 is a [neural network](https://en.wikipedia.org/wiki/Neural_network): a differentiable map $f: \mathcal{X} \rightarrow \mathbb{R}^K$.
# Note, in particular, that the network outputs a vector of real numbers.
# These are typically transformed to probabilities by way of the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) $\sigma$.
# In other words, for input $x$, $\hat{y} = \sigma(f(x))$ is a probability vector of size $K$.
# The $k$-th element of this vector is the "belief" that the network assigns to $x$ being a member of class $k$.
#
# Given a set of observations $\mathcal{D} = \{(x_1, y_1), \ldots, (x_N, y_N)\}$, the cross-entropy loss for this task is
#
# $$
# L(\mathcal{D}) = \frac{1}{N}\sum_{n}H(p_{n},q_{n})
# $$
#
# where $q_{n}=\sigma(f(x_{n}))$ and $p_{n}$ is the probability mass
# function which places all of its mass on $y_{n}$.
# Expanding this, we obtain what is to some the more familiar representation
#
# $$
# L(\mathcal{D}) = -\frac{1}{N}\sum\_{n}[\log\sigma(f(x\_{n}))]\_{y_{n}}.
# $$
#
# ## See also
#
# * PyTorch [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# * Keras [CategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)
