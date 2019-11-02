---
layout: post
title: All of Statistics - Chapter 1 Solutions
date: 2020-04-30 12:00:00-0800
tags: all-of-statistics
---

**Acknowledgements**: Thanks to Ben S. for correcting some mistakes.

## 1.

Let $$i < j$$.
Since $$B_i \subset A_i$$ and $$B_j \cap A_i = \emptyset$$, it follows that $$B_i$$ and $$B_j$$ are disjoint.

Since $$A_1 \subset A_2 \subset \cdots$$, it follows that $$A_n = \cup_{i = 1}^n A_i$$ for each $$n$$.

Suppose $$\cup_{i = 1}^n B_i = A_n$$ for some $$n$$.
By the previous claim, it follows that

$$
\begin{equation}
  \cup_{i = 1}^{n + 1} B_i
  = A_n \cup B_{n+1}
  = \left( \cup_{i = 1}^n A_i \right) \cup \left( A_{n + 1} \setminus \left( \cup_{i = 1}^n A_i \right) \right)
  = \cup_{i = 1}^{n + 1} A_i.
\end{equation}
$$

Lastly, let $$A_1 \supset A_2 \supset \cdots$$ be monotone decreasing.
Noting that $$A_1^c \subset A_2^c \subset \cdots$$ is monotone increasing,

$$
\begin{equation}
  \mathbb{P}(\cap_n A_n)
  = 1 - \mathbb{P}(\cup A_n^c)
  = 1 - \lim_n \mathbb{P}(A_n^c)
  = \lim_n 1 - \mathbb{P}(A_n^c)
  = \lim_n \mathbb{P}(A_n).
\end{equation}
$$

## 2.

Since $$\mathbb{P}(\emptyset \cup \emptyset) = 2 \mathbb{P}(\emptyset)$$ by additivity, it follows that $$\mathbb{P}(\emptyset) = 0$$.

If $$A$$ is contained in $$B$$, then

$$
\begin{equation}
  \mathbb{P}(B)
  = \mathbb{P}(A \cup B)
  = \mathbb{P}(A \cup \left( B \setminus A \right))
  = \mathbb{P}(A) + \mathbb{P}(B \setminus A)
  \geq \mathbb{P}(A)
\end{equation}
$$

As an immediate consequence of the previous two claims, it follows that $$\mathbb{P}(A) \leq \mathbb{P}(\Omega) = 1$$.

Since $$\mathbb{P}(A) + \mathbb{P}(A^c) = \mathbb{P}(A \cup A^c) = \mathbb{P}(\Omega) = 1$$, it follows that $$\mathbb{P}(A) = 1 - \mathbb{P}(A^c)$$.

Lastly, we point out that by taking $$A_2 = A_3 = \cdots = \emptyset$$ in the countable additivity property (Axiom 3), we obtain finite additivity: $$\mathbb{P}(A_1 \cup A_2) = \mathbb{P}(A_1) + \mathbb{P}(A_2)$$ for any disjoint sets $$A_1$$ and $$A_2$$.

## 3.

### a)

Note that

$$
\begin{equation}
  B_n
  = \cup_{i = n}^\infty A_i
  \supset \cup_{i = n + 1}^\infty A_i
  = B_{n + 1}.
\end{equation}
$$

Similarly,

$$
\begin{equation}
  C_n
  = \cap_{i = n}^\infty A_i
  \subset \cap_{i = n + 1}^\infty A_i
  = C_{n + 1}.
\end{equation}
$$

### b)

$$\omega$$ is in $$\cap_n B_n$$
$$\iff$$ $$\omega$$ is in $$B_n$$ for each $$n$$
$$\iff$$ for each $$n$$, we can find $$i \geq n$$ such that $$\omega$$ is in $$A_i$$.

*Remark*. A shorthand for $$\cap_n B_n$$ is $$\limsup_n A_n$$.

### c)

$$\omega$$ is in $$\cup_n C_n$$
$$\iff$$ $$\omega$$ is in $$C_n$$ for some $$n$$
$$\iff$$ we can find $$n$$ such that $$\omega$$ is in $$A_i$$ for each $$i \geq n$$.

*Remark*. A shorthand for $$\cup_n C_n$$ is $$\liminf_n A_n$$.

## 4.

Note that

$$
\begin{align}
  \omega \in \left(\cup_i A_i\right)^c
  & \iff \omega \notin \cup_i A_i \\
  & \iff \omega \notin A_i \text{ for each } i \\
  & \iff \omega \in A_i^c \text{ for each } i \\
  & \iff \omega \in \cap_i A_i^c.
\end{align}
$$

Similarly,

$$
\begin{align}
\omega \in \left( \cap_i A_i \right)^c
  & \iff \omega \notin \cap_i A_i \\
  & \iff \omega \notin A_i \text{ for some } i \\
  & \iff \omega \in A_i^c \text{ for some } i \\
  & \iff \omega \in \cup_i A_i^c.
\end{align}
$$

## 5.

The sample space for the repeated coin flip experiment is $$\{H,T\}^{\mathbb{N}}$$: the set of all functions from the natural numbers to $$\{H,T\}$$.
Let $$X_n$$ be one if the $$n$$-th toss is heads and zero otherwise.
Then, the probability of stopping at the $$k$$-th toss is

$$
\begin{multline*}
  \mathbb{P}(X_1 + \cdots + X_{k - 1} = 1) \times \mathbb{P}(X_k = 1)
  = \binom{k-1}{1} p \left( 1 - p \right)^{k - 2} \times p \\
  = \left(k - 1\right) p^2 \left( 1 - p \right)^{k-2}.
\end{multline*}
$$

The above simplifies to $$(k - 1) 2^{-k}$$ in the case of a fair coin.

## 6.

Let $$\mathbb{P}$$ be a probability measure on $$\mathbb{N}$$.
By additivity, $$1 = \mathbb{P}(\mathbb{N}) = \sum_n \mathbb{P}(\{n\})$$.
Suppose $$\mathbb{P}$$ is uniform.
Then, $$\mathbb{P}(\{n\}) = c$$ for each $$n$$ and hence $$\mathbb{P}(\mathbb{N}) = c \cdot \infty$$ (we interpret $$0 \cdot \infty = 0$$), a contradiction.
 
## 7.

Define $$B_n$$ as in the hint.
By our findings in Questions 1 and 2,

$$
\begin{equation}
  \mathbb{P}(\cup_n A_n)
  = \mathbb{P}(\cup_n B_n)
  = \sum_n \mathbb{P}(B_n)
  \leq \sum_n \mathbb{P}(A_n).
\end{equation}
$$

## 8.

Since

$$
\begin{equation}
  \mathbb{P}(\cup_i A_i^c)
  \leq \sum \mathbb{P}(A_i^c)
  = 0,
\end{equation}
$$

it follows that

$$
\begin{equation}
  \mathbb{P}(\cap_i A_i)
  = 1 - \mathbb{P}( \left( \cap_i A_i \right)^c )
  = 1 - \mathbb{P}(\cup_i A_i^c)
  \geq 1 - 0
  = 1.
\end{equation}
$$

## 9.

First, note that $$\mathbb{P}(A \mid B) = \mathbb{P}(A \cap B) / \mathbb{P}(B) \geq 0$$.
In particular, $$\mathbb{P}(\Omega \mid B) = 1$$.
Lastly, let $$A_1, A_2, \ldots$$ be disjoint.
Then,

$$
\begin{equation}
  \mathbb{P}(\cup_n A_n \mid B)
  = \frac{\mathbb{P}(\cup_n \left( A_n \cap B \right))}{\mathbb{P}(B)}
  = \sum_n \frac{\mathbb{P}(A_n \cap B)}{\mathbb{P}(B)}
  = \sum_n \mathbb{P}(A_n \mid B).
\end{equation}
$$

## 10.

Without loss of generality, we can assume that the player picks door 1 and Monty reveals there is no prize behind door 2.
Then, the player is left between choosing door $$i = 1$$ or $$i = 3$$.
It follows that

$$
\begin{equation}
  p_i
  \equiv \mathbb{P}(\omega_1=i\mid\omega_2=2)
  = \frac{\mathbb{P}(\omega_2=2\mid\omega_1=i)\mathbb{P}(\omega_1=i)}{\mathbb{P}(\omega_2=2)}
  = \frac{\mathbb{P}(\omega_2=2\mid\omega_1=i)}{3\mathbb{P}(\omega_2=2)}.
\end{equation}
$$

In particular,

$$
\begin{equation}
  \mathbb{P}(\omega_2=2\mid\omega_1=i)
  = \begin{cases}
    1/2 & \text{if }i=1\\
    1 & \text{if }i=3.
  \end{cases}
\end{equation}
$$

Since the player should pick $$i$$ to maximize $$p_i$$, the player should switch from door 1 to door 3.

## 11.

First, note that that

$$
\begin{equation}
  \mathbb{P}(A^c \cap B^c)
  = \mathbb{P}(A^c) - \mathbb{P}(B) + \mathbb{P}(A \cap B).
\end{equation}
$$

Using the independence of $$A$$ and $$B$$,

$$
\begin{align}
  \mathbb{P}(A^c\cap B^c)
  & = \mathbb{P}(A^c) - \mathbb{P}(B) + \mathbb{P}(A) \mathbb{P}(B) \\
  & = \mathbb{P}(A^c) - \left( 1 - \mathbb{P}(A) \right) \mathbb{P}(B) \\
  & = \mathbb{P}(A^c) - \mathbb{P}(A^c) \mathbb{P}(B) \\
  & = \mathbb{P}(A^c) \left( 1 - \mathbb{P}(B) \right) \\
  & = \mathbb{P}(A^c) \mathbb{P}(B^c).
\end{align}
$$

## 12.

Let $$G_0$$ (respectively, $$G_1$$) be the event that the side of the seen (respectively, unseen) card is green.
Since $$\mathbb{P}(G_0) = 1/3 + 1/3 \cdot 1/2 = 1/2$$,
Then,

$$
\begin{equation}
  \mathbb{P}(G_1 \mid G_0)
  = \frac{\mathbb{P}(G_0 \cap G_1)}{\mathbb{P}(G_0)}
  = \frac{1/3}{1/2}
  = 2/3.
\end{equation}
$$

## 13.

### a)

The sample space for this question is identical to that of Question 5.

### b)

We stop at the third toss if and only if the first three flips are $$HHT$$ or $$TTH$$.
If $$p$$ is the probability of heads, then the probability of this is $$p^2 (1 - p) + (1 - p)^2 p = p (1 - p)$$.
In the case of a fair coin, this simplifies to $$1 / 4$$.

## 14.

Let $$A$$ and $$B$$ be events.

Suppose $$\mathbb{P}(A) = 0$$.
Then, $$\mathbb{P}(A \cap B) \leq \mathbb{P}(A) = 0$$ and hence $$\mathbb{P}(A \cap B) = 0 = \mathbb{P}(A) \mathbb{P}(B)$$.

Suppose $$\mathbb{P}(A) = 1$$.
Then, $$\mathbb{P}(A^c) = 0$$ and hence by our most recent findings, $$A^c$$ and $$B^c$$ are independent.
By our findings in Question 11, it follows that $$A$$ and $$B$$ are independent.

Suppose now that $$A$$ is independent of itself.
Then, $$\mathbb{P}(A) = \mathbb{P}(A\cap A) = \mathbb{P}(A)\mathbb{P}(A)$$ and hence either $$\mathbb{P}(A) = 0$$ or $$\mathbb{P}(A) = 1$$.

## 15.

Let $$B_k$$ be an indicator random variable that is one if and only if the $$k$$-th child has blue eyes.
Let $$B = B_1 + B_2 + B_3$$.
Let $$p = 1/4$$ be the probability of having blue eyes and $$q = 1 - p$$.

### a)

Note that

$$
\begin{equation}
  \mathbb{P}(B \geq 2 \mid B \geq1)
  = \frac{\mathbb{P}(B \geq 2)}{\mathbb{P}(B \geq 1)}
  = \frac{1 - \mathbb{P}(B \leq 1)}{1 - \mathbb{P}(B = 0)}.
\end{equation}
$$

Moreover, $$\mathbb{P}(B = 0) = q^3$$ and $$\mathbb{P}(B = 1) = 3pq^2$$.
Therefore,

$$
\begin{equation}
  \mathbb{P}(B \geq 2 \mid B \geq1)
  = \frac{1 - q^3 - 3pq^2}{1 - q^3}
  = \frac{10}{37}.
\end{equation}
$$

### b)

Note that

$$
\begin{multline}
  \mathbb{P}(B\geq2\mid B_1 = 1)
  = \frac{\mathbb{P}(B_1=1,B_2+B_3\geq1)}{\mathbb{P}(B_1=1)}
  = \mathbb{P}(B_2+B_3\geq1)  \\
  = 1 - \mathbb{P}(B_2 + B_3 = 0)
  = 1 - q^2
  = \frac{7}{16}.
\end{multline}
$$

## 16.

Let $$A$$ and $$B$$ be events with $$\mathbb{P}(B)>0$$.
$$\mathbb{P}(A \cap B) = \mathbb{P}(A \mid B)\mathbb{P}(B)$$ follows by multiplying by $$\mathbb{P}(B)$$ on both sides of the definition of conditional probability.
Moreover, if $$A$$ and $$B$$ are independent,

$$
\begin{equation}
  \mathbb{P}(A\mid B)
  = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}
  = \frac{\mathbb{P}(A)\mathbb{P}(B)}{\mathbb{P}(B)}
  = \mathbb{P}(A).
\end{equation}
$$

## 17.

Assuming $$\mathbb{P}(BC)$$ and $$\mathbb{P}(C)$$ are positive, the result follows from combining

$$
\begin{equation}
  \mathbb{P}(ABC)
  = \frac{\mathbb{P}(ABC)}{\mathbb{P}(BC)} \mathbb{P}(BC)
  = \mathbb{P}(A \mid BC) \mathbb{P}(BC)
\end{equation}
$$

and

$$
\begin{equation}
  \mathbb{P}(BC)
  = \frac{\mathbb{P}(BC)}{\mathbb{P}(C)}\mathbb{P}(C)
  = \mathbb{P}(B \mid C) \mathbb{P}(C).
\end{equation}
$$

## 18.

If $$A_1, \ldots, A_k$$ are a partition of the sample space, then $$1 = \mathbb{P}(\cup_i A_i) = \sum_i \mathbb{P}(A_i)$$.
Moreover, for any event $$B$$,

$$
\begin{equation}
  \mathbb{P}(B)
  = \mathbb{P}(\left( \cup_i A_i \right) \cap B)
  = \mathbb{P}(\cup_i \left(A_i \cap B \right))
  = \sum_i \mathbb{P}(A_i \cap B).
\end{equation}
$$

If $$\mathbb{P}(B) > 0$$, then we can divide both sides by $$\mathbb{P}(B)$$ to get $$1 = \sum_i \mathbb{P}(A_i \mid B)$$.
Combining this with a previous equality, we get $$\sum_i\mathbb{P}(A_i) = \sum_i\mathbb{P}(A_i\mid B)$$.
Suppose now $$\mathbb{P}(A_1\mid B) < \mathbb{P}(A_1)$$.
Then,

$$
\begin{equation}
  \sum_{i \neq 1}\mathbb{P}(A_i)
  < \sum_{i \neq 1}\mathbb{P}(A_i \mid B).
\end{equation}
$$

It follows that $$\mathbb{P}(A_i \mid B) > \mathbb{P}(A_i)$$ for at least one $$i$$.

## 19.

We use $$M$$, $$W$$, and $$L$$ to denote the event that the user uses Mac, Windows, and Linux, respectively.
We use $$V$$ to denote the event that the user has the virus.

$$
\begin{multline}
  \mathbb{P}(W\mid V)
  = \frac{\mathbb{P}(V\mid W)\mathbb{P}(W)}{\mathbb{P}(V)}
  = \frac{\mathbb{P}(V\mid W)\mathbb{P}(W)}{\sum_{X\in\{M,W,L\}}\mathbb{P}(V\mid X)\mathbb{P}(X)}\\
  = \frac{82\times50}{65\times30+82\times50+50\times20}
  = \frac{82}{141}
  \approx 0.58.
\end{multline}
$$

## 20.

### a)

$$
\begin{equation}
  \mathbb{P}(C_i \mid H)
  = \frac{p_i \mathbb{P}(C_i)}{\mathbb{P}(H)}
  = \frac{p_i \mathbb{P}(C_i)}{\sum_j p_j \mathbb{P}(C_j)}
  = \frac{p_i}{\sum_j p_j}.
\end{equation}
$$

### b)

$$
\begin{equation}
  \mathbb{P}(H_2 \mid H_1)
  = \frac{\mathbb{P}(H_1 \cap H_2)}{\mathbb{P}(H_1)}
  = \frac{\sum_i p_i^2 \mathbb{P}(C_i)}{\sum_i p_i \mathbb{P}(C_i)}
  = \frac{\sum_i p_i^2}{\sum_i p_i}.
\end{equation}
$$

### c)

$$
\begin{equation}
  \mathbb{P}(C_i \mid B_4)
  = \frac{\mathbb{P}(C_i \cap B_4)}{\mathbb{P}(B_4)}
  = \frac{\mathbb{P}(B_4 \mid C_i) \mathbb{P}(C_i)}{\sum_j \mathbb{P}(B_4 \mid C_j) \mathbb{P}(C_j)}
  = \frac{\left( 1 - p_i \right)^3 p_i}{\sum_j \left(1 - p_j \right)^3 p_j}.
\end{equation}
$$

## 21.

TODO (Computer Experiment)

## 22.

TODO (Computer Experiment)

## 23.

TODO (Computer Experiment)
