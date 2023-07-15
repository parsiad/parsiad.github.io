---
date: 2023-04-28 12:00:00-0800
layout: post
title: Test if the sample mean is statistically significantly different from zero
hidden: true
---
Let $X_{1},\ldots,X_{N}$ be IID. The plug-in estimator for the mean is the sample mean $\bar{X}=(X_{1}+\cdots+X_{N})/N$.
The standard error of this estimator is $\sqrt{\operatorname{Var}(X_{1})/N}$.
Therefore, a normal confidence interval for the mean is
\begin{equation}
\bar{X}\pm c\sqrt{\frac{\operatorname{Var}(X_{1})}{N}}.
\end{equation}
In short, if we want to test if the mean is statistically significantly different from zero (assuming normality), the number of samples needs to satisfy
\begin{equation}
N>\frac{c^{2}\operatorname{Var}(X_{1})}{\left|\bar{X}\right|^{2}}.
\end{equation}
