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

# Lecture 14 (Logistic Regression and Generalised Models)

## Hypothesis Function
$$h_\theta(x)=\displaystyle\frac{1}{y_i-e^{-\theta^Tx}}$$
We use *gradient ascent* instead of *gradient descent* since $$LL(\theta)$$ is concave.

## Generalised Linear Models
We compute $P(y_i;\eta)$ where $\eta$ is a function of $\theta,x_i$ and $y_i\eta$ belongs to exponential family distribution.
$$P(y_i;\eta)=b(y)\exp\left(\eta y-a(\eta_i)\right)$$
(HW: Prove Bernoulli and Normal are special cases of the above equation)
The log-liklihood is given by:
$$LL(\eta)=\sum_{i=1}^m\left(\log(b(y_i))+\eta y_i-a(\eta_i)\right)$$
$$\implies \nabla_\theta(LL(\eta))=\sum_{i=1}^m\nabla_\theta(\eta_i)\left(y_i-a'(\eta_i)\right)$$
Following assumptions are made:

1. $\eta_i$ is a linear function of $x_i$
2. $h_\theta(x)=E[y|x;\theta]=g(\eta)$
3. $g^{-1}(\phi)$ is linearly dependent on $x$

