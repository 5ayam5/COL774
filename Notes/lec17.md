---
geometry:
- top=25mm
- left=20mm
- right=20mm
- bottom=30mm
documentclass: extarticle
fontsize: 12pt
numbersections: true
---

# Lecture 17 (Naive Bayes)

*Note: $y=b$ anywhere in this lecture is for $b=\{0, 1\}$*

It is a generative model

## Assumptions
1. $y\sim \text{Bernoulli}(\phi)$ ($y$ is usually considered to be discrete)
1. $x$ is discrete taking values $1, 2, \ldots, L$
1. $x_i\perp x_j|y$, that is, all $x_i$ are independent given $y$
1. (repeat) **Only the conditional probability is independent**
1. This is the only assumption made by naive Bayes

## Mathematical Analysis
$$P(x|y)=\prod_{i=1}^n{P(x_i|y)}$$
$$P(y|x)=\displaystyle\frac{P(x|y)P(y)}{P(x)} = \displaystyle\frac{\prod_{i=1}^n{P(x_i|y)}}{\sum_y{\prod_{i=1}^n{P(x_i|y)}}}$$
Now, to find the class, we can directly compute:
$$\underset{y}{argmax}P(y|x) = \underset{y}{argmax}P(x|y)P(y)$$

## Generating the Model
$$x_{j|y=b} = \text{Multinoulli}(\theta_{j|y=b})$$
$$\theta_{j|y=b} = \left(\theta_{j1|y=b},\ldots, \theta_{jL|y=b}\right)$$
In the above expression, $\displaystyle\sum_{l=1}^L{\theta_{jl|y=b}}=1$

Now we compute $argmax$ for $LL(\Theta)=\log\prod_{i=1}^m P(x_i, y_i; \Theta)$,
$$LL(\Theta) = \sum_{i=1}^m\left(\log P(y_i; \phi) + \log P(x_i|y_i; \Theta)\right)$$
The second term can be written as:
$$\sum_{j=1}^n\log P((x_i)_j|y; \Theta)$$
Now the above is *simplified* by adding probabilities for $y=1$ and $y=0$ and writing $P((x_i)_j|y; \Theta)$ and product of probabilities for each component.

(The solving has been skipped for sanity purposes)

On computing $\nabla_\Phi LL(\theta)=0$, we get:
$$\phi = \displaystyle\frac{\sum_{i=1}^m{y_i}}{m}$$
$$\theta_{jl|y=b} = \displaystyle\frac{\sum_{i=1}^m{(y_i=b)(x_j=l)}}{\sum_{i=1}^m{y_i=b}}$$

### Gaussian Naive Bayes Model
The assumptions made are that $x_j$ is independent and $x_j|y=b$ follows normal distribution. This gives $\Sigma$ as a diagonal matrix of $\sigma_{jb}$.
