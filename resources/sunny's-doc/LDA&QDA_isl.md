# Why don't we use Logistic Regression instead of Linear Discriminant Analysis

There are several reasons:
- When there are a substantial seperation between classes, the coefficents in logistic regression are suprisingly unstable. This method does not suffer from this problem

- If the distribution of $X$ is approximately normal in each class and the sample size is small, then this approach may be more accurate than logistic regression.

- The methods in this section can be naturally extend to more than 2 classes. In the case of more than 2 classes, we can also use multivariate logistic regression.

# Bayes's Theorem:

Reminder:
$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$

In the classification problem, we intend to estimate the probability that a point $x$ fall into a class $k$:
$$P(Y = k | X = x)$$
Based on Bayes's Theorem, we have:
$$P(Y = k | X = x) = \frac{P(X = x | Y = k) P(Y = k)}{P(X = x)}$$
Let's remap our terms:
- $P(X = x | Y = k) = f_k(x)$
- $P(Y = k) = \pi_k$
- $P(X = x) = \sum_{l = 1}^{K}\pi_k\cdot f_k(x)$
- $P(Y = y | X = x) = P_k(x)$

Rewrite the equation:
$$P_k(x) =  \frac{\pi_k f_k(x)}{\sum_{l = 1}^{K}\pi_l f_l(x)} $$

The above equation is the essential of Linear Discriminant Analysis and Quadratic Discriminant Analysis.

Let's break down the equation into 2 parts:
- The $\pi_k$
- The $f_k(x)$

As we know about these two terms from the remap, term $\pi_k$ stands for $P(Y = k)$ and $f_k(x)$ stands for $P(X = x | Y = k)$. Clearly, estimating $\pi_k$ is easy if we have the sample from the population, we simply compute the fraction of the trainning observations that belongs to the $k$-th class. However, estimating $f_k(x)$ is much more challenging. As we will see, to estimate $f_k(x)$, we typically have to make some assumption.