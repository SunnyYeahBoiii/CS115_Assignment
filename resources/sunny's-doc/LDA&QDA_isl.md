# Why don't we use Logistic Regression instead of Linear Discriminant Analysis

There are several reasons:
- When there are a substantial seperation between classes, the coefficents in logistic regression are suprisingly unstable. This method does not suffer from this problem

- If the distribution of $X$ is approximately normal in each class and the sample size is small, then this approach may be more accurate than logistic regression.

- The methods in this section can be naturally extend to more than 2 classes. In the case of more than 2 classes, we can also use multinomial logistic regression.

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

# Linear Discirminant Analysis for $p = 1$
For now, let's assume that $p = 1$, means that we only have 1 feature in our observations. In order to decide which class does $x$ belongs to, we will estimate $f_k(x)$ and plug into the equation to find $P_k(x)$ and then conclude that $x$ belongs to the class that has the highest $P_k(x)$.

In particular, we will assume that $f_k(x)$ is ***normal*** or ***Gaussian***. In the one dimensional settings, the normal density function takes the form:

$$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k} exp(-\frac{1}{2\sigma_k^2}(x - \mu_k)^2)$$

where $\mu_k$ and $\sigma^2_k$ are the mean and variance parameters for te $k$-th class. Please notice that $\pi_k$ denotes the prior probability that observation belongs to $k$-th class, not to be confused with $\pi \approx 3.14159$.

For now, let's further assume that $\sigma_1^2 = ... = \sigma_K^2$, that is, there is a shared variance term across $K$ classes, for simplicity, we can denote by $\sigma^2$. Substitute $f_k(x)$ into the equation.

$$P_k(x) = \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)}{\sum_{l = 1}^K\pi_l\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_l)^2)}$$

To find the class where $x$ belongs, we need to find class $k$ whereas $P_k(x)$ is maximal, which means:

$$\delta_k(x) = \argmax_k \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)}{\sum_{l = 1}^K\pi_l\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_l)^2)}$$

$$\Leftrightarrow \delta_k(x) = \argmax_k \pi_k \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)$$

$$\Leftrightarrow \delta_k(x) = \argmax_k \pi_k  exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)$$


$$\Leftrightarrow \delta_k(x) = \argmax_k \log(\pi_k) + \frac{x\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2}$$

The terms Linear Discriminant Analysis is from the fact that the discriminant function $\delta_k(x)$ is in linear form.