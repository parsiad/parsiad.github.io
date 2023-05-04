---
date: 2023-05-02 12:00:00-0800
layout: post
title: Uncorrelated coordinates in PCA
---
Let $X$ be an $N\times p$ real matrix.
Let $D$ be a real diagonal matrix, $w$ be a vector of nonnegative weights that sum to one, and
\begin{equation}
Y=\left(X-\frac{1}{N}ee^{\intercal}X\right)D^{-1}.
\end{equation}
$Y$ is a demeaned and rescaled version of $X$.

Let $V_{k}$ be a real $p\times k$ matrix consisting of the first $k$ principal components of $Y$.
Just as we can interpret the rows of $Y$ as $p$-dimensional points $y_{1},\ldots,y_{N}$, we can interpret the rows $a_{1},\ldots,a_{N}$ of $YV_{k}$ as $k$-dimensional points in "PCA space".
A property of PCA space is that the coordinates are uncorrelated:
\begin{equation}
V_{k}^{\intercal}Y^{\intercal}YV_{k}-\frac{1}{N}V_{k}^{\intercal}Y^{\intercal}ee^{\intercal}YV_{k}=\Lambda_{k}-0=\Lambda_{k}
\end{equation}
where $\Lambda_{k}$ is the diagonal matrix consisting of the first $k$ eigenvalues of $Y^{\intercal}Y$.

Let $W_{k}=DV_{k}$. While $V_{k}$ is orthonormal, $W_{k}^{\intercal}W_{k}$ is generally dense.
However, similarly to the above, the rows $b_{1},\ldots,b_{N}$ of the matrix $XW_{k}$ have uncorrelated coordinates: 
\begin{equation}
W_{k}^{\intercal}X^{\intercal}XW_{k}
=W_{k}^{\intercal}\left(YD+\frac{1}{N}ee^{\intercal}X\right)^{\intercal}\left(YD+\frac{1}{N}ee^{\intercal}X\right)W_{k}
 =V_{k}^{\intercal}Y^{\intercal}YV_{k}+\frac{1}{N}W_{k}^{\intercal}X^{\intercal}ee^{\intercal}XW_{k}
 =\Lambda_{k}+\frac{1}{N}W_{k}^{\intercal}X^{\intercal}ee^{\intercal}XW_{k}.
\end{equation}
Note, in particular, that because $X$ is not demeaned, the second term on the last equation is not necessarily equal to zero.
