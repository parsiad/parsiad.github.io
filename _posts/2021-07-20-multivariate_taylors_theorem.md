---
date: 2021-07-20 12:00:00-0800
layout: post
redirect_from:
  - /blog/2021/multivariate-taylors-theorem/
title: Multivariate Taylor's Theorem
---
## Motivation

This short post derives Taylor's Theorem for multivariate functions by employing Taylor's Theorem in a single variable.

## Derivation

Let $f : \mathbb{R}^d \rightarrow \mathbb{R}$.
For vectors $x$ and $v$ in $\mathbb{R}^d$, define $g : \mathbb{R} \rightarrow \mathbb{R}$ by $g(t) = f(x + tv)$.
If $g$ is $K$ times differentiable at zero, Taylor's theorem in 1d tells us
\begin{equation}\label{eq:1d}\tag{1}
    f(x + tv)
    = g(t)
    = \sum_{k = 0}^K \frac{t^k}{k!} g^{(k)}(0)
    + o(t^K) \text{ as } t \rightarrow 0.
\end{equation}
Suppose
\begin{equation}\label{eq:derivative}\tag{2}
    g^{(k)}(t)
    = \sum_{i_1, \ldots, i_k}
      v_{i_1} \cdots v_{i_k}
      \frac{\partial^k f}{\partial x_{i_1} \cdots x_{i_k}}(x + tv).
\end{equation}
By chain rule,
\begin{equation}
    g^{(k + 1)}(t)
    = \sum_{i_1, \ldots, i_k}
      v_{i_1} \cdots v_{i_k}
      \left \langle
          v,
          \nabla \left[
              \frac{\partial^k f}{\partial x_{i_1} \cdots x_{i_k}}
          \right] (x + tv)
      \right \rangle.
\end{equation}
Simplifying, we arrive at \eqref{eq:derivative} with $k$ replaced by $k + 1$.
Since \eqref{eq:derivative} is trivially satisfied at $k = 1$, it follows by induction that \eqref{eq:derivative} holds for all positive integers $k$.

The form \eqref{eq:derivative} is redundant since, assuming the conditions of [Clairaut's theorem](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives#Sufficiency_of_twice-differentiability), partial derivatives commute (e.g., $f_{x_1 x_2} = f_{x_2 x_1}$).
For a multi-index $\alpha = (\alpha_1, \ldots, \alpha_d)$ in $\mathbb{Z}^d_{\geq 0}$, define $|\alpha| = \alpha_1 + \cdots + \alpha_d$ and
\begin{equation}
    D^\alpha f
    = \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1} \cdots \partial x_d^{\alpha_d}}.
\end{equation}
With this notation we can write \eqref{eq:derivative} as
\begin{equation}
    g^{(k)}(t)
    = \sum_{|\alpha| = k}
      \frac{k!}{\alpha_1! \cdots \alpha_d!}
      v_1^{\alpha_1} \cdots v_d^{\alpha_d}
      D^\alpha f(x + tv).
\end{equation}
Substituting this into \eqref{eq:1d}, we obtain the desired Taylor polynomial:
\begin{equation}
    f(x + tv)
    = \sum_{k = 0}^K t^k
      \sum_{|\alpha| = k}
      \frac{1}{\alpha_1! \cdots \alpha_d!}
      v_1^{\alpha_1} \cdots v_d^{\alpha_d}
      D^\alpha f(x)
      + o(t^K) \text{ as } t \rightarrow 0
\end{equation}

## Remainder

If, in addition, $g$ is $K + 1$ times differentiable, we can extend the Cauchy or Lagrange form of the remainder term to the multivariate setting.
For example, the Lagrange form is
\begin{equation}
    o(t^K) = \frac{t^{K + 1}}{\left( K + 1 \right)!} g^{(K + 1)}(\theta)
\end{equation}
where $\theta$ is some number between zero and $t$.
Substituting \eqref{eq:derivative} into the above, we can obtain by triangle inequality, the (loose) bound
\begin{equation}
    o(t^K)
    \leq \frac{
             \left( d \left| t \right| \left \Vert v \right \Vert_\infty \right)^{K + 1}
         }{\left( K + 1 \right)!}
         \max_{|\alpha| = K + 1}
         \left |D^\alpha f(x + \theta v) \right|.
\end{equation}
