---
geometry:
- top=25mm
- left=20mm
- right=20mm
- bottom=30mm
documentclass: extarticle
fontsize: 12pt
numbersections: true
title: Lecture 19 (Support Vector Machines)
--- 

# Mathematical Setup
1. $y\in\{-1, 1\}$
1. $w^Tx+b=0$ is the equation of the hyperplane
1. Not all examples contribute to the model, we try to find the "support vectors" from the training set
1. SVMs find hyperplanes which maximise the minimum margin (margin defined below)

## Margin of Point
(normalised) Signed distance of point from hyperplane, it is given by:
$$\displaystyle\frac{w^Tx^i+b}{||w||_2}$$

## Max-Margin Based Classifier (SVMs property)
$$\underset{\gamma, w, b}\max(\gamma: \forall i, \gamma^i\geq\gamma)$$
$$\gamma^i = y^i\times\left(\displaystyle\frac{w^Tx^i+b}{||w||_2}\right)$$
$$\hat{\gamma}=||w||\gamma$$
$$\displaystyle\underset{\gamma, w, b}\max(\frac{\hat{\gamma}}{||w||}: \forall i, y^i\times(w^Tx^i+b)\geq\hat{\gamma})$$
Suppose $\hat{\gamma}^*, w^*, b^*$ is an optimal solution, then any multiple of these is also an optimal solution.
$$\displaystyle\implies 1, \frac{w^*}{\hat{\gamma}^*}, \frac{b^*}{\hat{\gamma}^*}\text{ is also an optimal solution}$$
$$\implies 1, w^{'*}, b^{'*}$$
$$\displaystyle\implies\underset{w, b}\max(\frac{1}{||w||}: \forall i, y^i\times(w^Tx^i+b)\geq 1)$$
$$\displaystyle\implies\underset{w, b}\max(\frac{1}{||w||^2}: \forall i, y^i\times(w^Tx^i+b)\geq 1)$$
$$\displaystyle\implies\underset{w, b}\min(\frac{1}{2}w^Tw: \forall i, y^i\times(w^Tx^i+b)\geq 1)$$
1. This is the optimisation problem for SVMs
1. It is a subproblem of convex (constrained) optimisation problem

# Constrained Optimisation Problem
$$\underset{w}\min(f(w): (g_i(w)\leq 0, \forall i\in\{1,\ldots, m\})\wedge(h_l(w)=0, \forall l\in\{1,\ldots, k\}))$$

## Convex Constrained Optimisation
1. $f$ is convex function
1. $g_i$ are convex functions
1. $h_l$ are affine, i.e., $h_l(w)=w^Tx+b$ (almost linear function, it allows for intercept term too)
