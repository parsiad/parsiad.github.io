---
layout: post
title: Multivariate Taylor's Theorem
date: 2021-07-20 12:00:00-0800
description: Derivation of Taylor's Theorem for multivariate functions.
---

## Motivation

This short post derives Taylor's Theorem for multivariate functions by employing Taylor's Theorem in a single variable.

## Derivation

Let $$f:\mathbb{R}^{d}\rightarrow\mathbb{R}$$. For a vectors $$x$$ and $$v$$ in $$\mathbb{R}^{d}$$, define $$g:\mathbb{R}\rightarrow\mathbb{R}$$ by $$g(t)=f(x+tv)$$.
If $$g$$ is $$K$$ times differentiable at zero, Taylor's theorem in 1d tell us

$$
\begin{equation}\label{eq:1d}\tag{1}
    f(x+tv)=g(t)=\sum_{k=0}^{K}\frac{t^{k}}{k!}g^{(k)}(0)+o(t{}^{K})\ \text{as }t\rightarrow0.
\end{equation}
$$

Suppose

$$
\begin{equation}\label{eq:gk}\tag{2}
    g^{(k)}(t)=\sum_{i_{1},\ldots,i_{k}}v_{i_{1}}\cdots v_{i_{k}}\frac{\partial f}{\partial x_{i_{1}}\cdots\partial x_{i_{k}}}(x+tv).
\end{equation}
$$

By chain rule,

$$
    g^{(k+1)}(t)=\sum_{i_{1},\ldots,i_{k}}v_{i_{1}}\cdots v_{i_{k}}\left\langle v,\nabla\left[\frac{\partial f}{\partial x_{i_{1}}\cdots\partial x_{i_{k}}}\right](x+tv)\right\rangle .
$$

Simplifying, we arrive at (\ref{eq:gk}) with $$k$$ replaced by $$k+1$$.
Since (\ref{eq:gk}) is satisfied at $$k=1$$, it follows by induction that (\ref{eq:gk}) holds for all integers $$k\geq1$$.

The form (\ref{eq:gk}) is redundant since, assuming the conditions of [Clairaut's theorem](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives#Sufficiency_of_twice-differentiability), partial derivatives commute (e.g., $$f_{x_{1}x_{2}}=f_{x_{2}x_{1}}$$).
For a multi-index $$i=(i_{1},\ldots,i_{d})$$ in $$\mathbb{Z}_{\geq0}^{d}$$, define $$|i|=i_{1}+\cdots+i_{d}$$ and

$$
    D^{i}f=\frac{\partial^{|i|}f}{\partial x_{i_{1}}\cdots\partial x_{i_{d}}}.
$$

With this notation we can write (\ref{eq:gk}) as

$$
    g^{(k)}(t)=\sum_{|i|=k}\frac{k!}{i_{1}!\cdots i_{d}!}v_{1}^{i_{1}}\cdots v_{d}^{i_{d}}D^{i}f(x+tv).
$$

Substituting this into (\ref{eq:1d}),

$$
    f(x+tv)
    =\sum_{k=0}^{K}t^{k}\sum_{|i|=k}\frac{1}{i_{1}!\cdots i_{d}!}v_{1}^{i_{1}}\cdots v_{d}^{i_{d}}D^{i}f(x)+o(t{}^{K})\ \text{as }t\rightarrow0,
$$

we obtain Taylor's Theorem for multivariate functions.

## Remainder

If, in addition, $$g$$ is $$K + 1$$ times differentiable, we can extend the Cauchy or Lagrange form of the remainder term to the multivariate setting.
For example, the Lagrange form is

$$
    o(t^{K})=\frac{t^{K+1}}{\left(K+1\right)!}g^{(K+1)}(\theta)
$$

where $$\theta$$ is some number between zero and $$t$$.
Substituting (\ref{eq:gk}) into the above, we can obtain, by triangle inequality, the (loose) bound

$$
o(t^{K})\leq\frac{\left(d\left|t\right|\left\Vert v\right\Vert _{\infty}\right)^{K+1}}{\left(K+1\right)!}\max_{|i|=K+1}\left|D^{i}f(x+\theta v)\right|.
$$
