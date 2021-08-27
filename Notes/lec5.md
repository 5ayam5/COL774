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

# Lecture 5 (Linear Regression)

## Mathematical Formulation
The equation for linear regression is:
$$y = \theta_1 \cdot x + \theta_0 + \epsilon$$
where, $h_\theta(x) = \theta_1 \cdot x + \theta_0$

The more general version when $x \in \mathbb{R}^n$ is:
$$h_\theta(x) = \sum_{i=0}^{n}{\theta_i \cdot x_i} = \theta^T \cdot x_i$$
($x$ is $n+1$ dimensional with $x_0 = 1$)

## How is $h_\theta$ Formulated?
Loss function:
$$J(\theta) = \frac{1}{2 \cdot m}\cdot \sum_{i=1}^{m}{(y_i - h_\theta(x_i))^2} = \frac{1}{m}\cdot \sum_{i=1}^{m}{(y_i - \theta^T \cdot x_i)^2}$$
Why is $abs$ not taken instead of square? This is because $abs$ is not differentiable at some points and squared error has "natural" justification (sir didn't elaborate much about this)

## How do we find $\theta$?
We try to compute
$$\underset{\theta \in \mathbb{R}^n}{argmin}(J(\theta))$$
$J(\theta)$ is quadratic in $\theta$.

