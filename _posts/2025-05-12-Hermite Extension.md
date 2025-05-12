---
title: "Extended Properties of Hermite Polynomials"
layout: distill
tags: math
categories: ML-Theory
featured: true

mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Lizhong Zheng
    url: "https://lizhongzheng.github.io/"
    affiliations:
      name: EECS, MIT


bibliography: blog_references.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: The Starting Question
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Definitions and Notations
  
 
---

## The Starting Question
Most of the results in this post are taken from this paper by T. P. Davis <d-cite key="TPDavis2024"></d-cite>. It is really a hidden gem, and I keep a local copy [here](/assets/pdf/DavisTP2024.pdf). The reason that I get interested in this paper is because there are some really nice properties of the Hermite polynomials listed in this paper in a very clear way. In particular, as we discussed in the previous [post](https://lizhongzheng.github.io/blog/2025/Hermite/), Hermite polynomials defines an orthonormal basis with respect to a given Gaussian distribution $\mathcal N(\mu, \sigma^2)$ as the reference distribution. As we change the reference distribution to a different Normal distribution, what was a set of orthonormal functions become no longer orthonormal. The question we have in mind reading the paper by Davis is to figure out how sensitive is the orthogonality to the change of reference distribution.  

## Definitions and Notations

There are several versions of the [Hermite Polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials) defined in the literature. Unfortunately, we won't use any of them, but will define our own before making connections to the more widely used notations. 

To start, we denote $${\mathcal N}(x; \mu, \sigma^2)$$ as the Gaussian probability density function (pdf) with mean $\mu$ and variance $\sigma^2$ for variable $x$. We often call this distribution a **reference distribution**.

---
Definition: Hermite Basis Functions
: For a given reference distribution as a Gaussian density function $${\mathcal N}(x; \mu, \sigma^2)$$, the Hermite polynomials are the sequence of polynomials $\phi_0, \phi_1, \ldots, \phi_k, \ldots$, where $\phi_k(\cdot; \mu, \sigma^2): \mathbb {R \longrightarrow R}$ is a $k$-degree polynomial, with parameter $\mu, \sigma^2$, and satisfy

\begin{equation}
\label{eqn:Hermite}
\langle \phi_i(\cdot;\mu, \sigma^2), \phi_j(\cdot; \mu, \sigma^2) \rangle \stackrel{\Delta}{=} \int_{-\infty}^\infty \; {\mathcal N} (x; \mu, \sigma^2) \cdot \phi_i(x; \mu, \sigma^2) \cdot \phi_j (x; \mu, \sigma^2) \; dx = \delta_{ij}
\end{equation}

For the special case that $\mu=0, \sigma^2=1$, we use a short-hand notation 

$$
\begin{align*}
\phi_k (x) \stackrel{\Delta}{=} \phi_k(x; 0, 1), \qquad \forall x, k
\end{align*}
$$

The collection of functions $\{\phi_k(\cdot), k=1, 2, \ldots\}$ form an orthonormal basis w.r.t. the standard Normal distribution $\mathcal {N}(0,1)$.

---

There is a simple variable change when we change the reference distribution in our notation:

\begin{equation}
\label{eqn:change_parameter}
\phi_k(x; \mu, \sigma^2) = \phi_k\left( \frac{x-\mu}{\sigma}\right)
\end{equation}

For the rest of this post, we will try to use the standard Normal distribution as the reference. To simplify notation, we denote

$$
\omega(x) = \mathcal{N}(x; 0, 1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
$$


In the literature, the "probabilist's Hermite polynomials" are defined as

$$ He_k (x) = (-1)^k e^{\frac{x^2}{2}} \frac{d^k}{dx^k} e^{-\frac{x^2}{2}}$$

Most of the known properties of the Hermite polynomials are presented in this form. Our notation is related to the commonly used ones in the following form. 

$$\phi_k(x) \stackrel{\Delta}{=} \frac{1}{\sqrt{k!}} \cdot He_k(x) = \frac{1}{\sqrt{k!}} \cdot (-1)^k \frac{1}{\omega(x)} \frac{d^k \omega(x)}{dx^k} $$


## The Question

We see that we need to make changes to the Hermit polynomials when we change the reference distribution. A natural questions is, how non-orthogonal the Hermite basis are when we take the orthonormal basis around one Normal distribution to a different Normal distribution. 


We would like to evaluate the inner product, with $\mathcal{N}(0,1)$ as the reference,

$$
\langle \phi_i(\cdot; \mu, \sigma^2), \phi_j(\cdot; \mu, \sigma^2) \rangle
$$

or closely related

$$
\langle \phi_i(\cdot; \mu, \sigma^2), \phi_j(\cdot) \rangle = \int \omega(x) \cdot \phi_i(x; \mu, \sigma^2) \cdot \phi_j(x) \; dx
$$

This question is closely related to the concept of "connection" in [Amari's works](https://bookstore.ams.org/mmono-191). Here, we would like to proceed with an algebraic solution. 

## Hermite Polynomial Expansion

We start with some useful properties. 

Definition: Square Integrable Functions w.r.t. the Gaussian Weighting Function
: 
We say a function $f: \mathbb{R} \to \mathbb {R}$ is square integrable w.r.t. the standard Gaussian weighting function $\omega(x)$, denoted as $f \in L_2(\mathbb {R}, \omega(x)dx)$, if 

$$
\int_{-\infty}^\infty |f(x)|^2 \cdot \omega(x) \, dx < \infty
$$

and 

$$
\lim_{x \to \pm \infty} f(x)\cdot \omega(x) = 0
$$


---
Theorem 1: 
: 
If $f \in L_2(\mathbb{R}, \omega(x)dx)$, then we have 

$$
f(x) = \sum_k  d_k \cdot \phi_k(x) , \qquad x \in\mathbb{R}
$$

where 

$$
\begin{align*}
d_k &= \langle f, \phi_k\rangle = \int_{-\infty}^\infty \omega(x) \cdot f(x) \cdot \phi_k(x) \, dx\\
&= \frac{1}{\sqrt{k!}} \int_{-\infty}^\infty \omega(x) \cdot \frac{\partial^k f(x)}{\partial x^k} \, dx
\end{align*}
$$


{% details Proof: %}
The expansion follows from the fact that $\phi_k$'s form an orthonormal basis. The only thingwe need is to connect the two ways to compute $d_k$, which can be derived by repeatedly using integration by part: 

$$
\begin{align*}
d_k &=  \int_{-\infty}^\infty \omega(x) \cdot f(x) \cdot \phi_k(x) \, dx\\
&= \int_{-\infty}^\infty \omega(x) \cdot f(x) \cdot \frac{(-1)^k}{\sqrt{k!}} \cdot \frac{1}{\omega(x)} \cdot \frac{d^k \omega(x)}{dx^k} \, dx \\
&= \frac{1}{\sqrt{k!}} \left[(-1)^k \frac{d^{k-1} w(x)}{dx^{k-1}} \cdot f(x)\Big|_{-\infty}^\infty + (-1)^{k-1} \int_{-\infty}^\infty \frac{d^{k-1} w(x)}{dx^{k-1}}  \frac{d f(x)}{dx} \,dx\right]\\
& \ldots \\
&= \frac{1}{\sqrt{k!}} \int_{-\infty}^\infty \omega(x)\cdot  \frac{d^k f(x)}{dx^k} \,dx
\end{align*}
$$

{% enddetails%}

---
Theorem 2: Appell Sequence
: 
The Hermite basis functions satisfy the following recurrence relation: 

$$
\begin{align*}
\frac{d}{dx} \phi_k(x) &= \sqrt{k} \cdot \phi_{k-1}(x)\\
\frac{d}{dx} \left(\omega(x) \cdot \phi_k(x)\right) &= - \sqrt{k+1} \cdot \left(\omega(x) \cdot \phi_{k+1}(x)\right)
\end{align*}
$$

---

We do not prove this Theorem as it is readily available from many [sources](https://lizhongzheng.github.io/blog/2025/Hermite/). However, there is a nice corollary as follows. 

<div class="corollary-box" markdown="1">
  <strong>Corollary 3: Taylor Expansion</strong>  
  For $m\leq k$, 

$$
\begin{align*}
\phi_k^{(m)}(x) &= \sqrt{\frac{k!}{(k-m)!}} \cdot \phi_{k-m}(x)\\
\phi_k(x) &= \sum_{m=0}^k \frac{1}{\sqrt{m!}} \sqrt{k\choose m} \cdot \phi_{k-m}(0) \cdot x^m 
\end{align*}
$$

</div>

The first equation follows from repeatedly applying the recursion in Theorem 2. The second comes from the Taylor expansion of $\phi_k(x)$ at $x=0$. The constants $\frac{1}{\sqrt{n!}} \phi_n(0)$ are known as the **Hermite Numbers**. They are non-zero if and only if $n$ is even. The corollary says that as long as we know all the Hermite numbers, we can find the coefficients for all the Hermite polynomials.

<div class="corollary-box" markdown="1">
  <strong>Corollary 4: Shift by Scalar</strong>  

$$
\begin{align*}
\phi_k(a x + b)  &= \sum_{m=0}^k \frac{1}{\sqrt{m!}} \sqrt{k\choose m} \cdot \phi_{k-m}(ax) \cdot b^m
\end{align*}
$$

</div>

{% details Proof:%}

We take the Taylor expansion to function $s \mapsto \phi_k(a s)$ at $s=x$: 

$$
\begin{align*}
\phi_k(a x + b) & = \phi_k \left(a\left(x+\frac{b}{a}\right)\right) \\
&= \sum_{m=0}^\infty \frac{1}{m!} \phi_k^{(m)}(as)\Big|_{s=x} \cdot \left(\frac{b}{a}\right)^m\\
&= \sum_{m=0}^k \frac{1}{m!} \sqrt{\frac{k!}{(k-m)!}} \cdot \phi_{k-m}(ax) \cdot a^{m}\left(\frac{b}{a}\right)^m
\end{align*}
$$


{% enddetails%}

---
Theorem 5: Hermite Multiplication Theorem
: 
The Hermite expansion of $\phi_n(\gamma x)$ is 

\begin{equation}
\label{eqn:multiplication_thm}
\phi_n(\gamma x) = \sum_{k=0}^{\lfloor \frac{n}{2}\rfloor} \sqrt{\frac{n!}{(n-2k)!}} \cdot \frac{1}{2^k k!} (\gamma^2-1)^k \gamma^{n-2k} \cdot \phi_{n-2k}(x)
\end{equation}

---