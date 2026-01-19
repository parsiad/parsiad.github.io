+++
template = "cv.html"
title = "CV"
+++

|                        |                                                     |
|------------------------|-----------------------------------------------------|
| **Email**              | `MY_FIRST_NAME.MY_LAST_NAME@gmail.com`              |
| **Website**            | [parsiad.ca](https://parsiad.ca)                    |
| **GitHub**             | [github.com/parsiad](https://github.com/parsiad.ca) |
| **Citizenship**        | Canadian                                            |
| **Work Authorization** | Green Card                                          |
| **Languages**          | English (native), Farsi (working)                   |

I am a researcher and engineer with deep expertise in machine learning, computer science, mathematics, and statistics.
I currently work on pretraining at OpenAI.

## Experience

**Member of Technical Staff, OpenAI** — 2025–Present  

- Pretraining research

**Quantitative Researcher, PDT Partners** — 2019–2025  

- Machine/statistical learning α research, trading research
- Built high-performance infrastructure for neural nets, gradient-boosted trees, Bayesopt

**Software Engineer, Google LLC** — 2018–2019

- Developed GPU/TPU-friendly TensorFlow Probability ODE solvers (see this [BDF solver](https://www.tensorflow.org/probability/api_docs/python/tfp/math/ode/BDF) and, for an application, an [FFJORD demo](https://www.tensorflow.org/probability/examples/FFJORD_Demo))
- Researched and implemented novel auction mechanisms with substantive PNL impact

**Research Assistant Professor, University of Michigan Department of Mathematics** — 2018

- Continued Ph.D. research topics (see below)
- Taught [MATH 525: Probability Theory](https://parsiad.ca/MATH-525)

## Education

**Ph.D., University of Waterloo** — 2017

- Research on stochastic control, HJBQVI, numerical linear/multilinear algebra
- Published in top applied mathematics journals (SIAM, AMS)
- Awards totaling CAD 100k
- Thesis — [Impulse Control in Finance: Numerical Methods and Viscosity Solutions](https://arxiv.org/pdf/1712.01647.pdf)

**MMath, University of Waterloo** — 2013

**BSc, Computer Science & Mathematics, Simon Fraser University** — 2011

## Selected Publications

Anything older than 2015 is excluded.
For a more complete list, I refer you to [my Google Scholar page](https://scholar.google.ca/citations?hl=en&user=PGB51pwAAAAJ&view_op=list_works&sortby=pubdate).

- P. Azimzadeh, ["A zero-sum stochastic differential game with impulses, precommitment, and unrestricted cost functions"](https://doi.org/10.1007/s00245-017-9445-x), In *Appl. Math. Optim.*, vol. 79, no. 2, pp. 483–514, 2019.
- P. Azimzadeh, E. Bayraktar, ["High order Bellman equations and weakly chained diagonally dominant tensors"](https://epubs.siam.org/doi/abs/10.1137/18M1196923), In *SIAM J. Matrix Anal. Appl.*, vol. 40, no. 1, pp. 276-298, 2019.
- P. Azimzadeh, ["A fast and stable test to check if a weakly diagonally dominant matrix is a nonsingular M-matrix"](https://doi.org/10.1090/mcom/3347), In *Math. Comp.*, vol. 88, no. 316, pp. 783–800, 2019.
- M. Amy, P. Azimzadeh, M. Mosca, ["On the CNOT-complexity of CNOT-phase circuits"](https://doi.org/10.1088/2058-9565/aad8ca), In *Quantum Science and Technology*, 2018.
- P. Azimzadeh, E. Bayraktar, G. Labahn, ["Convergence of implicit schemes for Hamilton-Jacobi-Bellman quasi-variational inequalities"](https://doi.org/10.1137/18M1171965), In *SIAM J. Control and Optim.*, vol. 56, no. 6, pp. 3994–4016, 2018.
- P. Azimzadeh, P. A. Forsyth, ["Weakly chained matrices, policy iteration, and impulse control"](https://doi.org/10.1137/15M1043431), In *SIAM J. Numer. Anal.*, vol. 54, no. 3, pp. 1341–1364, 2016.
- P. Azimzadeh, T. Carpenter, ["Fast Engset computation"](https://doi.org/10.1016/j.orl.2016.02.011), In *Oper. Res. Lett.*, vol. 44, no. 3, pp. 313–318, 2016.
- P. Azimzadeh, P. A. Forsyth, ["The existence of optimal bang-bang controls for GMxB contracts"](https://doi.org/10.1137/140953885), In *SIAM J. Financial Math.*, vol. 6, no. 1, pp. 117-139, 2015.

## Peer Review

SIAM Journal on Control and Optimization, SIAM Journal on Numerical Analysis, Mathematics of Operations Research, European Journal of Operations Research, Applied Mathematics and Optimization, Journal of Computational Finance, Journal of Combinatorics and Optimization, Dynamic Games and Applications, Numerical Algorithms, IEEE Transactions on Automatic Control  

## Some (F)OSS Contributions

Though I do not have much time to contribute to (F)OSS, some small contributions can be seen below. Please keep in mind that some of these are old and not indicative of current engineering practices:

- Started the [TensorFlow Probability ODE Suite](https://www.tensorflow.org/probability/api_docs/python/tfp/math/ode)
- Maintained [GNU Octave Financial](https://octave.sourceforge.io/financial/index.html) and made its SDE framework (see [my blog post](https://parsiad.ca/blog/sdes-and-monte-carlo-in-octave-financial))
- Made [mlinterp](https://parsiad.ca/mlinterp), a C++ header-only template meta-programming library for linear interpolation in ℝᵈ
- Made [QuantPDE](https://parsiad.ca/QuantPDE), a C++ HJB and HJBQVI solver used in papers and by other academics
- Made [fast-engset](https://github.com/parsiad/fast-engset), a Python package to compute quantities in the Engset model
- Made [lazy-table](https://github.com/parsiad/lazy-table), a Python package to pretty-print tables from generators
- Merged [Optuna PR2503](https://github.com/optuna/optuna/pull/2503) using an O(N log N) algorithm for computing an efficient frontier in ℝ²
