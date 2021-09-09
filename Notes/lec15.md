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

# Lecture 15 (GDA)

## Preliminiaries
$$\prod P(x_i, y_i;\theta)=\prod P(y_i;\theta) P(x_i|y_i;\theta)$$
We assume $x|y$ to be a normal distribution

Covariance matrix is:
$$\Sigma_{ij}=cov(X_i, X_j)=E\left[(X-E[X])(X-E[X])^T\right]$$

Now, we consider $X$ to be normally distributed as $N(\mu_X, \Sigma)$. Also note that $\Sigma$ will be symmetric and positive semi-definitive

$$P(x=z) = \displaystyle\frac{1}{\sqrt{2\pi |\Sigma|}}\exp\left(-\displaystyle\frac{(X-\mu)^T\Sigma^{-1}(X-\mu)}{2}\right)$$

## Gaussian Discriminant Analysis
The idea is to generate the contour of $x|y=0$ and $x|y=1$ using $\mu_0, \Sigma_0$ and $\mu_1, \Sigma_1$ respectively.

$$\Theta = \left(\phi, \mu_0, \Sigma_0, \mu_1, \Sigma_1\right)$$
$\Theta$ is the set of parameters of our model. Now, $LL(\Theta)$$ is given by
$$LL(\Theta) = \sum_{i=1}^m y_i\log(\phi) + (1-y_i)\log(1-\phi) + y_i\left(\log\displaystyle\frac{1}{\sqrt{2\pi |\Sigma_0|}} - \displaystyle\frac{(x_i-\mu_0)^T\Sigma_0^{-1}(x_i-\mu_0)}{2}\right)$$ $$+ (1-y_i)\left(\log\displaystyle\frac{1}{\sqrt{2\pi |\Sigma_1|}} - \displaystyle\frac{(x_i-\mu_1)^T\Sigma_1^{-1}(x_i-\mu_1)}{2}\right)$$

Now, $\nabla_\Theta LL(\Theta)=0$ gives,
$$\phi=\displaystyle\frac{\mu_Y}{m}$$
$$\mu_0=\displaystyle\frac{\sum_{i=1}^m{(1-y_i)x_i}}{\sum_{i=1}^m{1-y_i}}$$
$$\Sigma_0=\displaystyle\frac{\sum_{i=1}^m{(1-y_i)(x_i-\mu_0)(x_i-mu_0)^T}}{\sum_{i=1}^m{1-y_i}}$$
$$\mu_1=\displaystyle\frac{\sum_{i=1}^m{y_ix_i}}{\sum_{i=1}^m{y_i}}$$
$$\Sigma_1=\displaystyle\frac{\sum_{i=1}^m{y_i(x_i-\mu_1)(x_i-mu_1)^T}}{\sum_{i=1}^m{y_i}}$$

