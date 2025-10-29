# Table of contents:
1. [Why don't we use Logistic Regression instead of Linear Discriminant Analysis](#why)
2. [Bayes's Theorem](#bayes)
3. [Linear Discriminant Analysis for $p = 1$](#lda-p1)
4. [Linear Discriminant Analysis for $p > 1$](#lda-p2)

# Why don't we use Logistic Regression instead of Linear Discriminant Analysis <a id = "why"></a>

There are several reasons:
- When there are a substantial seperation between classes, the coefficents in logistic regression are suprisingly unstable. This method does not suffer from this problem

- If the distribution of $X$ is approximately normal in each class and the sample size is small, then this approach may be more accurate than logistic regression.

- The methods in this section can be naturally extend to more than 2 classes. In the case of more than 2 classes, we can also use multinomial logistic regression.

# Bayes's Theorem: <a id = "bayes"></a>

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

# Linear Discirminant Analysis for $p = 1$ <a id = "lda-p1"></a>
For now, let's assume that $p = 1$, means that we only have 1 feature in our observations. In order to decide which class does $x$ belongs to, we will estimate $f_k(x)$ and plug into the equation to find $P_k(x)$ and then conclude that $x$ belongs to the class that has the highest $P_k(x)$.

In particular, we will assume that $f_k(x)$ is ***normal*** or ***Gaussian***. In the one dimensional settings, the normal density function takes the form:

$$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma_k} exp(-\frac{1}{2\sigma_k^2}(x - \mu_k)^2)$$

where $\mu_k$ and $\sigma^2_k$ are the mean and variance parameters for te $k$-th class. Please notice that $\pi_k$ denotes the prior probability that observation belongs to $k$-th class, not to be confused with $\pi \approx 3.14159$.

For now, let's further assume that $\sigma_1^2 = ... = \sigma_K^2$, that is, there is a shared variance term across $K$ classes, for simplicity, we can denote by $\sigma^2$. Substitute $f_k(x)$ into the equation.

$$P_k(x) = \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)}{\sum_{l = 1}^K\pi_l\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_l)^2)}$$

To find the class where $x$ belongs, we need to find class $k$ whereas $P_k(x)$ is maximal, which means:

$$\underset{k}{argmax } \frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)}{\sum_{l = 1}^K\pi_l\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_l)^2)}$$

$$= \underset{k}{argmax } \pi_k \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)$$

$$= \underset{k}{argmax } \pi_k  exp(-\frac{1}{2\sigma^2}(x - \mu_k)^2)$$


$$= \underset{k}{argmax } \log(\pi_k) + \frac{x\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2}$$

Therefore, the discriminant function can be written as follow:

$$\delta_k(x) = \log(\pi_k) + \frac{x\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2}$$

$$\Leftrightarrow \delta_k(x) = \underbrace{x\frac{\mu_k}{\sigma^2}}_{\text{coefficent of } x} - \underbrace{\frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)}_{\text{constant term}}$$


The terms Linear Discriminant Analysis is from the fact that the discriminant function $\delta_k(x)$ is in linear form.

# Linear Discriminan Analysis for $p > 1$ <a id = "lda-p2"></a>
We will now extend the LDA classifier to the case of multiple predictor. To do this, we will say that $X = (X_1 , X_2 , ... , X_p)$ is drawn from a ***multi-variate normal*** or ***multi-variate Gaussian*** distribution, with a class-specific mean vector $\mu$ and a common covariance matrix $\Sigma$. Furthermore, we have :

$$X \sim \mathcal{N}(\mu , \Sigma)$$
$$\mu = E[X] = E[(X_1 , X_2 , ... , X_p)]$$
$$\Sigma_{i , j} = E[(X_i - \mu_i)(X_j - \mu_j)] = \text{Cov}(X_i , X_j)$$

Formally, the multivariate Gaussian density is define as

$$f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1} (x - \mu_k))$$

Plugging into the LDA equation and perform some algebra, we will see that the Bayes classifier assigns $X = x$ to the class for which

$$\delta_k(x) = \mu_k^T \Sigma^{-1}x - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log(\pi_k)$$

is the largest. As we can see, this is the vector/matrix version of the original LDA for $p = 1$.

# Quadratic Discriminant Analysis 
