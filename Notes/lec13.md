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

# Lecture 13 (Classification Problem)

## Classification
1. Given $\{x_i, y_i\}_{i=1}^m$ where $y_i$ is discrete and *currently* takes only values $0, 1$
2. The equation of separator is given as $\theta^Tx=0$
3. The ditribution is Bernoulli, i.e., $P(y_i=1|x_i;\theta)=\Phi(\theta^Tx_i)$ ($\theta^Tx_i$ gives the normal distance of $x_i$ from the separator)
4. $\Phi(z)=\displaystyle\frac{1}{1+e^{-z}}$
5. Log liklihood ($LL(\theta)$):
$$\log(\prod_{i=1}^mP(y_i \text{ is predicted correctly}|x_i;\theta)) = \sum_{i=1}^m \left((y_i=1)\log(\Phi(\theta^Tx)) + (y_i=0)\log(1-\Phi(\theta^Tx))\right)$$
6. $\theta_{ML}=\underset{\theta}{argmax}(LL(\theta))$
7. This can now be solved using gradient descent, and $\nabla_\theta(LL(\theta)) = \sum_{i=1}^m \left(y_i-\displaystyle\frac{1}{1+e^{-\theta^Tx_i}}\right)x_i$

