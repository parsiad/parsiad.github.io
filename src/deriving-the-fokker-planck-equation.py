# %% [raw]
# +++
# aliases = [
#   "/blog/2024/deriving_the_fokker_planck_equation"
# ]
# date = 2024-01-19
# title = "Deriving the Fokker-Planck equation"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# ## Motivation
#
# The Fokker-Planck equation is a [partial differential equation](https://en.wikipedia.org/wiki/Partial_differential_equation) (PDE) which describes the evolution of the [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) of an [Ito diffusion](https://en.wikipedia.org/wiki/It%C3%B4_diffusion).
# Since it is a PDE, it admits solutions in certain special cases and is amenable to [numerical methods for PDEs](https://en.wikipedia.org/wiki/Partial_differential_equation#Numerical_solutions) in the general case.
#
# ## Derivation
#
# Consider the SDE
#
# $$
# \mathrm{d}X_{t}=a(t,X_{t})\mathrm{d}t+b(t,X_{t})\mathrm{d}W_{t}
# $$
#
# with bounded coefficients: (i.e., $\sup_{t,x}|a(t,x)|+\sup_{t,x}|b(t,x)|<\infty$).
# This requirement is used below to apply Fubini's theorem but can be relaxed.
#
# Let $f:\mathbb{R}\rightarrow\mathbb{R}$ be smooth with compact support.
# By Ito's lemma,
#
# $$
# \mathrm{d}f=f_{x}\mathrm{d}X_{t}+\frac{1}{2}f_{xx}\mathrm{d}X_{t}^{2}=\left(af_{x}+\frac{1}{2}b^{2}f_{xx}\right)\mathrm{d}t+bf_{x}\mathrm{d}W_{t}.
# $$
#
# Taking expectations of both sides,
#
# $$
# \mathbb{E}\left[\mathrm{d}f\right]=\mathbb{E}\left[\left(af_{x}+\frac{1}{2}b^{2}f_{xx}\right)\mathrm{d}t\right].
# $$
#
# The above is formalism for the expression
#
# $$
# \mathbb{E}f(X_{T})-\mathbb{E}f(X_{t})=\mathbb{E}\left[\int_{t}^{T}a(s,X_{s})f_{x}(X_{s})+\frac{1}{2}b(s,X_{s})^{2}f_{xx}(X_{s})\mathrm{d}s\right].
# $$
#
# By [Fubini's theorem](https://en.wikipedia.org/wiki/Fubini's_theorem), we can interchange the expectation and integral on the right hand side.
# Moreover, by the [Mean Value Theorem](https://en.wikipedia.org/wiki/Mean_value_theorem), we can find $\xi$ between $t$ and $T$ such that
#
# $$
# \frac{\mathbb{E}f(X_{T})-\mathbb{E}f(X_{t})}{T-t}=\mathbb{E}\left[a(\xi,X_{\xi})f_{x}(X_{\xi})+\frac{1}{2}b(\xi,X_{\xi})^{2}f_{xx}(X_{\xi})\right].
# $$
#
# Taking limits as $T\downarrow t$ and applying the [Dominated Convergence Theorem](https://en.wikipedia.org/wiki/Dominated_convergence_theorem),
#
# $$
# \frac{\partial}{\partial t}\left[\mathbb{E}f(X_{t})\right]=\mathbb{E}\left[a(t,X_{t})f_{x}(X_{t})+\frac{1}{2}b(t,X_{t})^{2}f_{xx}(X_{t})\right].
# $$
#
# Let $p(t,\cdot)$ be the density of $X_{t}$.
# Then, the above is equivalent to
#
# $$
# \frac{\partial}{\partial t}\int p(t,x)f(x)\mathrm{d}x=\int p(t,x)\left(a(t,x)f_{x}(x)+\frac{1}{2}b(t,x)^{2}f_{xx}(x)\right)\mathrm{d}x.
# $$
#
# Applying [integration by parts](https://en.wikipedia.org/wiki/Integration_by_parts) to the right hand side,
#
# $$
# \frac{\partial}{\partial t}\int f(x)p(t,x)\mathrm{d}x=\int f(x)\left(-\frac{\partial}{\partial x}\left[p(t,x)a(t,x)\right]+\frac{1}{2}\frac{\partial^{2}}{\partial x^{2}}\left[p(t,x)b(t,x)^{2}\right]\right)\mathrm{d}x.
# $$
#
# Since this holds for all functions $f$, it follows that
#
# $$
# \frac{\partial p}{\partial t}(t,x)=-\frac{\partial}{\partial x}\left[p(t,x)a(t,x)\right]+\frac{1}{2}\frac{\partial^{2}}{\partial x^{2}}\left[p(t,x)b(t,x)^{2}\right].
# $$
#
# This is the Fokker-Planck equation in one dimension.
# The derivation for multiple dimensions is similar.
