# %% [raw]
# +++
# aliases = [
#   "/blog/2016/octave_financial_0.5.0_released"
# ]
# date = 2016-02-02
# title = "Octave Financial 0.5.0 released"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# I am happy to announce the release of the GNU Octave Financial package version 0.5.0. This is the *first* release since I took on the role of maintainer.
#
# If you do not already have GNU Octave, you can [grab a free copy here](https://www.gnu.org/software/octave/download.html).
#
# To install the package, launch Octave and run the following commands:
#
# ```
# pkg install -forge io
# pkg install -forge financial
# ```
#
# Perhaps the most exciting addition in this version is the Monte Carlo simulation framework, which is significantly faster than its MATLAB counterpart. A brief tutorial (along with benchmarking information) are available in [a previous post](/blog/2015/sdes_and_monte_carlo_in_octave_financial). Other additions include Black-Scholes options and greeks valuation routines, implied volatility calculations, and general bug fixes. Some useful links for GNU Octave Financial are below:
#
# * [Home page](http://octave.sourceforge.net/financial/index.html)
# * [Documentation](http://octave.sourceforge.net/financial/overview.html)
# * [News](http://octave.sourceforge.net/financial/NEWS.html)
