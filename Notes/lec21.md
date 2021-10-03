---
geometry:
- top=25mm
- left=20mm
- right=20mm
- bottom=30mm
documentclass: extarticle
fontsize: 12pt
numbersections: true
title: Lecture 21 (KKT Conditions)
--- 

# Karush–Kuhn–Tucker Conditions
1. These conditions are necessary and sufficient for the the primal-dual equality
1. Gradient vanishes at optimal parameters
$$\nabla _vL(w, \alpha, \beta)|_{v^*}=0$$
for $v=w^*, \alpha^*, \beta^*$
1. Primal and dual feasibility exists
1. Complementary slackness
$$\alpha_ig_i(w^*) = 0, \forall i\in\{1, \ldots, m\}$$

# (new) SVM Objective
$$L(w, b, \alpha) = \frac{1}{2}w^Tw + \sum_{i=1}^m\alpha_i(1-y^i(w^Tx^i+b))$$
The SVM dual is given by:
$$\underset{\alpha, \alpha\geq 0}{\max}[\underset{w, b}{min}[L(w, b, \alpha)]]$$
Now, we compute the gradient of the Lagrangian:
$$\nabla_wL(w, b, \alpha) = w - \sum_{i=1}^m{\alpha_iy^ix^i}$$
Equating the gradient to $0$, we get:
$$w=\sum_{i=1}^m{\alpha_iy^ix^i}$$
Therefore we see that only *active* constraints play a role. Now we compute gradient wrt $b$ and equate it to $0$:
$$\nabla_bL(w, b, \alpha) = -\sum_{i=1}^m{\alpha_iy^i} = 0$$
Now substituting these two conditions and eliminating $w, b$ from $L(w, b, \alpha)$, we get:
$$w(\alpha) = \theta_D(\alpha) = \sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i, j=1}^m{\alpha_i\alpha_jy^iy^jx^{iT}x^j}$$
Now our dual objective is:
$$\underset{\alpha, \alpha\geq 0, \alpha^TY=0}{\max}[w(\alpha)]$$

1. We can now solve this using Block Coordinate Descent - Sequential Minimal Optimisation [ofc you have to read this by self and it won't be discussed :)]
1. This algo optimises two variables at a time (keeping the other fixed)
1. This simplifies it to a quadratic expression in a single variable by using the second constraint and we are left with the simple constraint of $\alpha_i\geq 0$
1. We now find $b$ as (solved by looking at the inequation of $g_i$):
$$\frac{-1}{2}[\underset{y^i=-1}\max(w^Tx^i) + \underset{y^i=1}\min(w^Tx^i)]$$
where $w=\sum_{i=1}^m{\alpha_iy^ix^i}$
