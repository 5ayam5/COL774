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

# Lecture 6 (Gradient Descent)

## Finding ${argmin} J(\theta)$ - Gradient Descent
The idea of *gradient descent* is used, since it might not always be possible to find **zeroes** of the gradient of $J(\theta)$. The algorithm is something like:

1. $\theta$ is initialised to a random value, call it $\theta^0$
2. Now, $\theta$ is updated as
$$\theta^{(t+1)} = \theta^t - \eta \cdot \frac{\partial f(\theta)}{\partial \theta}\bigg\vert_{\theta^t}$$
In $n+1$ dimensional space, it looks like:
$$\theta^{(t+1)} = \theta^t - \eta \cdot \nabla_\theta f(\theta^t) \vert_{\theta^t}$$
($\theta^k$ is a $n+1$ dimensional vector, $\nabla_\theta$ is computed by taking partial derivate for each term individually)

