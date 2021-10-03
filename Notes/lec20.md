---
geometry:
- top=25mm
- left=20mm
- right=20mm
- bottom=30mm
documentclass: extarticle
fontsize: 12pt
numbersections: true
title: Lecture 20 (Lagrangian)
--- 

# Formulating the Problem
$$L(w, \alpha, \beta) = f(w) + \sum_{i=1}^m{\alpha_ig_i(w)} + \sum_{l=1}^k{\beta_lh_l(w)}, \alpha_i\geq 0$$
Consider a point $w$ which is *feasible*, then
$$\underset{\alpha, \beta}\max(L(w, \alpha, \beta)) = f(w)$$
Also, consider a point $w$ which is not feasible, then
$$\displaystyle\underset{\alpha, \beta}\max(L(w, \alpha, \beta)) = \infty$$
Therefore, the *primal* problem can be written as solving:
$$\underset{w}{\min}(\underset{\alpha, \beta, \alpha\geq 0}{\max}(L(w, \alpha, \beta)))$$
$$\implies\underset{w}{\min}[\theta_P(w)]$$
where, $\theta_p(w)$ is the *primal* objective and the entire problem is called the *primal* problem. The constraints have been made simpler and have been absorbed. The *dual* problem is given as:
$$\underset{\alpha, \beta, \alpha\geq 0}{\max}[\underset{w}{\min}{L(w, \alpha, \beta)}]$$
$$\implies\underset{\alpha, \beta, \alpha\geq 0}{\max}[\theta_D(\alpha, \beta)]$$
The relation between $\theta_P(w)$ and $\theta_D(\alpha, \beta)$ is given as:
$$\underset{w}{\min}[\theta_P(w)]\geq\underset{\alpha, \beta, \alpha\geq 0}{\max}[\theta_D(\alpha, \beta)]$$
$$p^*\geq d^*$$
We are interested in finding the condition when $p^*=d^*$ so that we can solve the problem easily.

# (Sufficient) Conditions for Strong Duality ($p^*=d^*$)
1. Primal problem is convex, and
1. Slaters conditions are satisfied  
$\exists w: g_i(w)<0\ \forall i\in\{1, \ldots, m\}$ and $h_l(w)=0\ \forall l\in\{1, \ldots, k\}$

The $2\textsuperscript{nd}$ condition will be true for SVMs if the data is linearly separable (this is where the concept of support vector comes)
