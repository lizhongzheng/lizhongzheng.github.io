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

---



