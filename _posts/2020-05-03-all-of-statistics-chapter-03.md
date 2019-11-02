---
layout: post
title: All of Statistics - Chapter 3 Solutions
date: 2020-05-03 12:00:00-0800
tags: all-of-statistics
---

## 1.

Let $$X_n$$ be the number of dollars at the $$n$$-th trial.
Then,

$$
\begin{equation}
\mathbb{E}[X_{n + 1} \mid X_n]
= \frac{1}{2} \left( 2 X_n + \frac{1}{2} X_n \right)
= \frac{5}{4} X_n.
\end{equation}
$$

By the rule of iterated expectations, $$\mathbb{E} X_{n + 1} = (5 / 4) \mathbb{E} X_n$$.
By induction, $$\mathbb{E} X_n = (5 / 4)^n c$$.

## 2.

If $$\mathbb{P}(X = c) = 1$$, then $$\mathbb{E}[X^2] = (\mathbb{E} X)^2 = c^2$$ and hence $$\mathbb{V}(X) = 0$$.

The converse is more complicated.
We claim that whenever $$Y$$ is a nonnegative random variable, $$\mathbb{E}Y=0$$ implies that $$\mathbb{P}(Y=0)=1$$.
In this case, it is sufficient to take $$Y=(X-\mathbb{E}X)^{2}$$ to conclude that $$\mathbb{P}(X=\mathbb{E}X)=1$$.

To substantiate the claim, suppose $$\mathbb{E}Y=0$$. Take $$A_{n}=\{Y\geq1/n\}$$.
Then,

$$
\begin{equation}
0
=\mathbb{E}Y
=\mathbb{E}[YI_{A_{n}} + YI_{A_n^c}]
\geq\mathbb{E}[YI_{A_{n}}]
\geq \frac{1}{n}\mathbb{P}(A_{n}).
\end{equation}
$$

It follows that $$\mathbb{P}(A_{n})=0$$ for all $$n$$.
By continuity of probability,

$$
\begin{equation}
\mathbb{P}(Y>0)
=\mathbb{P}(\cup_{n}A_{n})
=\lim_{n}\mathbb{P}(A_{n})
=0.
\end{equation}
$$

## 3.

Since $$F_{Y_n}(y) = \mathbb{P}(X_1 \leq y)^n = y^n$$, it follows that $$f_{Y_n}(y) = n y^{n - 1}$$.
Therefore,

$$
\begin{equation}
\mathbb{E} Y_n
= n \int_0^1 y^n dy
= \frac{n}{n + 1}.
\end{equation}
$$

## 4.

Note that $$X_n = \sum_{i = 1}^n (1 - 2B_i) = n - 2 \sum_i B_i$$ where $$B_1, \ldots, B_n \sim \operatorname{Bernoulli}(p)$$ are IID.
It follows that $$\mathbb{E} X_n = n - 2 n \mathbb{E} B_1 = n - 2np$$ and $$\mathbb{V}(X_n) = 4 n \mathbb{V}(B_1) = 4 n p (1 - p)$$.

## 5.

Let $$\tau$$ be the number of tosses until a heads is observed.
Let $$C$$ denote the result of the first toss.
Then,

$$
\begin{equation}
\mathbb{E} \tau
= \frac{1}{2} \left( \mathbb{E}\left[\tau \mid C = H\right] + \mathbb{E}\left[\tau \mid C = T\right] \right)
= \frac{1}{2} \left( 1 + \left(1 + \mathbb{E} \tau\right) \right)
\end{equation}
$$

Solving for $$\mathbb{E} \tau$$ yields $$2$$.

## 6.

$$
\begin{multline}
\mathbb{E}[Y]
= \sum_y y \mathbb{P}(Y = y)
= \sum_y y \mathbb{P}(r(X) = y)
= \sum_y y \mathbb{P}(X \in r^{-1}(y)) \\
= \sum_y y \sum_{x \in r^{-1}(y)} \mathbb{P}(X = x)
= \sum_y \sum_{x \in r^{-1}(y)} r(x) \mathbb{P}(X = x)
= \sum_x r(x) \mathbb{P}(X = x)
\end{multline}
$$

## 7.

Integration by parts yields

$$
\begin{multline}
\mathbb{E}X
=\int_{0}^{\infty}xf_{X}(x)dx
=\lim_{y\rightarrow\infty}yF_{X}(y)-\int_{0}^{y}F_{X}(x)dx\\
=\lim_{y\rightarrow\infty}\int_{0}^{y}F_{X}(y)-F_{X}(x)dx
=\lim_{y\rightarrow\infty}\int_{0}^{\infty}\left(F_{X}(y)-F_{X}(x)\right)I_{(0,y)}(x)dx.
\end{multline}
$$

Define $$G_y(x)=(F_{X}(y)-F_{X}(x))I_{(0,y)}(x)$$.
Note that $$G_y$$ converges pointwise to $$1-F_{X}$$ as $$y\rightarrow\infty$$.
Moreover, $$y \mapsto G_y$$ is monotone increasing.
The desired result follows by Lebesgue's monotone convergence theorem.

## 8.

The first two claims follow from

$$
\begin{equation}
\mathbb{E} \overline{X}
= \frac{1}{n} \sum_i \mathbb{E} X_i
= \mathbb{E} X_1
\equiv \mu
\end{equation}
$$

and

$$
\begin{equation}
\mathbb{V}(\overline{X})
= \frac{1}{n^2} \sum_i \mathbb{V}(X_1)
= \frac{1}{n} \mathbb{V}(X_1)
\equiv \frac{\sigma^2}{n}.
\end{equation}
$$

As for the final claim, note that

$$
\begin{equation}
\left(n-1\right)S_{n}^{2}=\sum_{i}\left(X_{i}-\overline{X}\right)^{2}=\sum_{i}X_{i}^{2}-2X_{i}\overline{X}+\overline{X}^{2}
\end{equation}
$$

and hence

$$
\begin{equation}
\frac{n-1}{n}\mathbb{E}\left[S_{n}^{2}\right]=\mathbb{E}\left[X_{1}^{2}\right]-2\mathbb{E}\left[X_{1}\overline{X}\right]+\mathbb{E}\left[\overline{X}^{2}\right].
\end{equation}
$$

Next, note that $$\mathbb{E}[X_{1}^{2}]=\sigma^{2}+\mu^{2}$$ and $$\mathbb{E}[\overline{X}^{2}]=\sigma^{2}/n+\mu^{2}$$.
Moreover,

$$
\begin{equation}
X_{1}\overline{X}=\frac{1}{n}\left(X_{1}^{2}+X_{1}\sum_{j\neq1}X_{j}\right)
\end{equation}
$$

and hence $$\mathbb{E}[X_{1}\overline{X}]=\sigma^{2}/n+\mu^{2}$$.
Substituting these findings into the equation above yields $$\mathbb{E}[S_{n}^{2}]=\sigma^{2}$$, as desired.

## 9.

TODO (Computer Experiment)

## 10.

The MGF of a normal random variable is $$\exp(t^2 / 2)$$.
Therefore, $$\mathbb{E} \exp(X) = \sqrt{e}$$ and

$$
\begin{equation}
\mathbb{V}(\exp(X))
= \mathbb{E}[\exp(2X)] - (\mathbb{E} \exp(X))^2
= e^{2} - e.
\end{equation}
$$

## 11.

### a)

This was already solved in Question 4.

### b)

TODO (Computer Experiment)

## 12.

TODO

## 13.

### a)

Let $$C$$ denote the result of the coin toss.
Then,

$$
\begin{equation}
\mathbb{E} X
= \mathbb{E} \left[ \operatorname{Unif}(0, 1) I_{\{C = H\}} + \operatorname{Unif}(3, 4) I_{\{C = T\}} \right]
= \frac{1}{2} \left( \mathbb{E} \operatorname{Unif}(0, 1) + \mathbb{E} \operatorname{Unif}(3, 4) \right)
= 2.
\end{equation}
$$

### b)

Similarly to Part (a),

$$
\begin{equation}
\mathbb{E} \left[ X^2 \right]
= \frac{1}{2} \left( \mathbb{E} \left[ \operatorname{Unif}(0,1)^2 \right] + \mathbb{E} \left[ \operatorname{Unif}(3,4)^2 \right] \right)
= \frac{19}{3}.
\end{equation}
$$

Therefore, $$\mathbb{V}(X) = 19 / 3 - 4 = 7 / 3$$.

## 14.

The result follows from

$$
\begin{multline}
\operatorname{Cov}\left(\sum_{i}a_{i}X_{i},\sum_{j}b_{j}Y_{j}\right) \\
=\mathbb{E}\left[\left(\sum_{i}a_{i}X_{i}\right)\left(\sum_{j}b_{j}Y_{j}\right)\right]-\mathbb{E}\left[\sum_{i}a_{i}X_{i}\right]\mathbb{E}\left[\sum_{j}b_{j}Y_{j}\right]\\
=\sum_{i,j}a_{i}b_{j}\mathbb{E}\left[X_{i}Y_{j}\right]-\sum_{i,j}a_{i}b_{j}\mathbb{E}X_{i}\mathbb{E}Y_{j}
=\sum_{i,j}a_{i}b_{j}\left(\mathbb{E}\left[X_{i}Y_{j}\right]-\mathbb{E}X_{i}\mathbb{E}Y_{j}\right).
\end{multline}
$$

## 15.

First, note that $$\mathbb{V}(2X - 3Y + 8) = \mathbb{V}(2X - 3Y)$$.
Moreover,

$$
\mathbb{E}\left[\left(2X-3Y\right)^{2}\right]
=\int_{0}^{2}\int_{0}^{1}\left(2x-3y\right)^{2}\frac{1}{3}\left(x+y\right)dxdy
=\frac{86}{9}
$$

and

$$
\mathbb{E}\left[2X-3Y\right]
=\int_{0}^{2}\int_{0}^{1}\left(2x-3y\right)\frac{1}{3}\left(x+y\right)dxdy
=-\frac{23}{9}.
$$

Therefore, $$\mathbb{V}(2X - 3Y) = 245 / 81$$.

## 16.

In the (absolutely) continuous case,

$$
\begin{multline}
\mathbb{E}\left[r(X)s(Y)\mid X=x\right]
=\int r(x)s(y)f_{Y\mid X}(y\mid x)dy
=r(x)\int s(y)f_{Y\mid X}(y\mid x)dy\\
=r(x)\mathbb{E}\left[s(Y)\mid X=x\right].
\end{multline}
$$

Taking $$s = 1$$ yields $$\mathbb{E}[r(X) \mid X = x] = r(x)$$.
The discrete case is similar.
A more general notion of conditional expectation requires Radon-Nikodym derivatives.

## 17.

By the tower property,

$$
\mathbb{E}\left[\mathbb{V}(Y\mid X)\right]
=\mathbb{E}\left[\mathbb{E}\left[Y^{2}\mid X\right]-\mathbb{E}\left[Y\mid X\right]^{2}\right]
=\mathbb{E}\left[Y^{2}\right]-\mathbb{E}\left[\mathbb{E}\left[Y\mid X\right]^{2}\right]
$$

and

$$
\mathbb{V}(\mathbb{E}\left[Y\mid X\right])
=\mathbb{E}\left[\mathbb{E}\left[Y\mid X\right]^{2}\right]-\mathbb{E}\left[\mathbb{E}\left[Y\mid X\right]\right]^{2}
=\mathbb{E}\left[\mathbb{E}\left[Y\mid X\right]^{2}\right]-\mathbb{E}\left[Y\right]^{2}.
$$

The desired result follows from summing the two quantities.

## 18.

Since

$$
\begin{equation}
\mathbb{E}[XY]
=\mathbb{E}[\mathbb{E}[XY\mid Y]]
=\mathbb{E}[\mathbb{E}[X \mid Y] Y]
=\mathbb{E}[cY]
=c\mathbb{E}Y
\end{equation}
$$

and $$\mathbb{E}X =\mathbb{E}[\mathbb{E}[X\mid Y]] = c$$ by the tower property, $$\operatorname{Cov}(X,Y)=\mathbb{E}[XY] - \mathbb{E}X\mathbb{E}Y = 0$$.

## 19.

Unlike the distribution of $$X_1 \sim \operatorname{Unif}(0, 1)$$, the distribution of $$(X_1 + \cdots + X_n)/n$$ is concentrated around $$\mathbb{E}[X_1]$$.
As $$n$$ increases, so too does the concentration.

## 20.

For a vector $$a$$ with entries $$a_i$$,

$$
\begin{equation}
\mathbb{E}\left[a^{\intercal}X\right]
=\mathbb{E}\left[\sum_{j}a_{j}X_{j}\right]
=\sum a_{j}\mathbb{E}X_{j}
=a^{\intercal}\mathbb{E}X.
\end{equation}
$$

For a matrix $$A$$ with entries $$a_{ij}$$, define the column vector $$a_{i\star}$$ as the transpose of the $$i$$-th row of $$A$$.
Then,

$$
\begin{equation}
(\mathbb{E}\left[AX\right])_{i}
=\mathbb{E}\left[(AX)_{i}\right]
=\mathbb{E}\left[a_{i\star}^\intercal X\right]
=a_{i\star}^\intercal\mathbb{E}X.
\end{equation}
$$

Therefore, $$\mathbb{E}[AX]=A\mathbb{E}X$$.

Next, using our findings in Question 14,

$$
\begin{equation}
\mathbb{V}(a^{\intercal}X)
=\mathbb{V}\left(\sum_{j}a_{j}X_{j}\right)
=\sum_{i, j} a_{i}\operatorname{Cov}(X_{i},X_{j})a_{j}
=a^{\intercal}\mathbb{V}(X)a.
\end{equation}
$$

As before, we can generalize this to the matrix case by noting that

$$
(\mathbb{V}(AX))_{ij}
=\operatorname{Cov}((AX)_{i},(AX)_{j})
=\operatorname{Cov}(a_{i\star}^\intercal X,a_{j\star}^\intercal X)
=\sum_{k,\ell}a_{ik}\operatorname{Cov}(X_{k},X_{\ell})a_{j\ell}.
$$

Therefore, $$\mathbb{V}(AX)=A\mathbb{V}(X)A^{\intercal}$$.

## 21.

If $$\mathbb{E}[Y\mid X]=X$$, then

$$
\begin{equation}
\mathbb{E}[XY]
=\mathbb{E}[\mathbb{E}[XY\mid X]]
=\mathbb{E}[X\mathbb{E}[Y\mid X]]
=\mathbb{E}[X^{2}]
\end{equation}
$$

and $$\mathbb{E}Y=\mathbb{E}[\mathbb{E}[Y\mid X]]=\mathbb{E}X$$.
Therefore,

$$
\begin{equation}
\operatorname{Cov}(X,Y)
=\mathbb{E}[XY] - \mathbb{E}X \mathbb{E}Y
=\mathbb{E}[X^{2}]-(\mathbb{E}X)^{2}
=\mathbb{V}(X).
\end{equation}
$$

## 22.

### a)

Note that $$\mathbb{E}[YZ]=\mathbb{E}I_{(a,b)}(X)=b-a$$.
Moreover, $$\mathbb{E}Y=\mathbb{E}I_{(0,b)}(X)=b$$ and $$\mathbb{E}Z=\mathbb{E}I_{(a,1)}(X)=1-a$$.
Since $$\mathbb{E}[YZ]\neq\mathbb{E}Y\mathbb{E}Z$$, $$Y$$ and $$Z$$ are dependent.

### b)

If $$Z = 0$$, then $$X \leq a < b$$ and hence $$Y = 1$$.
Therefore, $$\mathbb{E}[Y \mid Z = 0] = 1$$ trivially.
Moreover,

$$
\begin{equation}
\mathbb{E}\left[Y\mid Z=1\right]
=\frac{\mathbb{E}\left[YZ\right]}{\mathbb{P}(Z=1)}
=\frac{b-a}{1-a}.
\end{equation}
$$

## 23.

Let $$K \sim \operatorname{Poisson}(\lambda)$$.
The MGF of $$K$$ is

$$
\begin{equation}
\mathbb{E}\left[e^{tK}\right]
=e^{-\lambda}\sum_{k}\frac{\lambda^{k}e^{tk}}{k!}
=e^{-\lambda}\sum_{k}\frac{\left(\lambda e^{t}\right)^{k}}{k!}
=\exp(\lambda\left(e^{t}-1\right))
\end{equation}
$$

Let $$X \sim N(\mu, \sigma^2)$$.
Then,

$$
\begin{multline}
\sigma\sqrt{2\pi}\mathbb{E}\left[e^{tX}\right]
=\int_{-\infty}^{\infty}\exp\left\{ -\frac{1}{2\sigma^{2}}\left(\left(x-\mu\right)^{2}-2t\sigma^{2}x\right)\right\} dx\\
=\exp\left(t\mu+t^{2}\sigma^{2}/2\right)\int_{-\infty}^{\infty}\exp\left\{ -\frac{1}{2\sigma^{2}}\left(x-\mu-t\sigma^{2}\right)^{2}\right\} dx.
\end{multline}
$$

Therefore, the MGF of $$X$$ is $$\exp(t\mu+t^{2}\sigma^{2}/2)$$.

Lastly, let $$Y \sim \operatorname{Gamma}(\alpha, \beta)$$.
Then,

$$
\begin{equation}
\mathbb{E}\left[e^{tY}\right]
=\beta^{\alpha}\int_{0}^{\infty}\frac{x^{\alpha-1}e^{(t-\beta)x}}{\Gamma(a)}dx
=\left(\frac{\beta}{t-\beta}\right)^{\alpha}\int_{0}^{\infty}\frac{\left(t-\beta\right)^{\alpha}x^{\alpha-1}e^{(t-\beta)x}}{\Gamma(a)}dx
\end{equation}
$$

is finite whenever $$t < \beta$$.
Therefore, under the same condition, the MGF of $$Y$$ is $$(1 - t / \beta)^{-\alpha}$$.

## 24.

Suppose $$\beta>t$$. Then,

$$
\begin{equation}
\mathbb{E}\left[\exp(tX_{1})\right]
=\int_{0}^{\infty}\beta\exp(\left(t-\beta\right)x)dx
=\frac{\beta}{\beta-t}
\end{equation}
$$

and hence

$$
\begin{equation}
\mathbb{E}\left[\exp\left(t\sum_{i}X_{i}\right)\right]
=\mathbb{E}\left[\exp\left(tX_{1}\right)\right]^{n}
=\left(\frac{\beta}{\beta-t}\right)^{n}
=\left(1-\frac{t}{\beta}\right)^{-n}.
\end{equation}
$$

Since this is the MGF of a Gamma distribution, it follows that the sum of IID exponentially distributed random variables are Gamma distributed.
