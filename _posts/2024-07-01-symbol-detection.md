---
title: "Symbol Detection in Wireless Communication"
layout: post
tags: H-Score wireless-communication 
categories: ML-Application
featured: true
authors:
 - name: Lizhong Zheng
 - name: Xiangxiang Xu
toc:
 beginning: true
 

---

> ### The Key Points
> In our previous developments, we introduced the [H-score network](https://lizhongzheng.github.io/blog/2024/H-Score/), which is a way to learn informative features from datasets, similar to a normal neural network, but was claimed to be more flexible. In this post, we use a concrete example in wireless communications to demonstrate this flexibility. In a nutshell, the flexibility comes from changing the objective from learning to solve a specific inference task to learning of feature functions that carry information that can be reused and combined with other sources of information. We demonstrate that this approach can help us to design neural-network-based solutions not to a single case, but a parameterized set of cases, as required by this specific engineering problem. We use this example to discuss the general methodology of integrating learning modules to engineering solutions. 


## Previously
This post is based on a sequence of previous posts. We briefly summarize the main points that we will be using here. 

In [Modal Decomposition](http://localhost:4000/blog/2024/modal-decomposition/), we stated that the dependence between two random variables, $$\mathsf{x, y}$$ can be decomposed into a sequence of correlation between feature pairs $$f_i(\mathsf{x}), g_i(\mathsf{y}), i=1, 2, \ldots$$. These feature functions are defined with the following optimization 

$$ 
\underline{f}^\ast, \underline{g}^\ast = \arg\min_{\substack{\underline{f} \in \mathcal {F_X}^k \\ \underline{g} \in \mathcal {F_Y}^k}} \; \Vert \mathrm{PMI}_{\mathsf{x,y}} -\underline{f} \otimes \underline{y}\Vert^2
$$ 

where $$\mathrm{PMI}_{\mathsf{x,y}} = \log \frac{P_{\mathsf{xy}}}{P_{\mathsf{x}} P_{\mathsf{y}}}$$ is the point-wise mutual information function; $$\underline{f} = [f_1, \ldots, f_k], \underline{g} = [g_1, \ldots, g_k]$$ are two collections of orthonormal feature functions  with correlation $$\sigma_i = \rho (f_i(\mathsf{x}), g_i(\mathsf{y}))$$ in a descending order.


|![test image](/assets/img/Hscorenetwork.png){: width="250" style="float:left; padding-right:30px"}|![test image](/assets/img/nested H2.png){: width="450" }|
|<b> H-Score Network </b>|<b> Nested H-Score Network to find features orthogonal to a given $$\bar{f}$$ </b>|

<br>

In [H-score](https://lizhongzheng.github.io/blog/2024/H-Score/), we defined a new metric called the H-score:

$$
\mathscr{H}(\underline{f}, \underline{g}) =\mathrm{trace} (\mathrm{cov} (\underline{f}, \underline{g})) - \frac{1}{2} \mathrm{trace}(\mathbb E[\underline{f} \cdot \underline{f}^T] \cdot \mathbb E[\underline{g} \cdot \underline{g}^T])
$$

and a network architecture called the H-score network, as shown in the figure, where we can learn the informative features defined in the modal decomposition by maximizing the H-score with a given dataset. 

In [nested H-score network](https://lizhongzheng.github.io/blog/2024/nested-H-score/), we introduced a nested architecture, see figure, to enforce orthogonality constraints in the learning process. 

The main points of this sequence of works is to introduce the key concepts and building blocks of **feature-centric learning**, where we view feature functions as the basic carrier of information and turn the learning objective from making a certain decision to finding feature functions that carry specific information. This general method makes the learning procedure more controllable and the learning results reusable. The H-score can be viewed as a quality metric for features. The nested H-score network can be viewed as a numerical method for the basic projection operation of features. Both are need in the further development. 


## A Wireless Communication Problem

|![test image](/assets/img/interferenceChannel.png){: width="350" }|
|<b> A Wireless Fading Interference Channel </b>|

<br>

The problem we work on is called "symbol detection in fading interference channel," which is a classical problem in wireless communications. For a more complete reference [here](https://stanford.edu/~dntse/wireless_book.html) is a wonderful textbook. Roughly the idea is depicted in the figure: a transmiter tries to send a symbol $$\mathsf {x}_1$$ to the receiver, but there is another transmitter, the interferer who is transmitting his signal at the same time. Thus, the receiver receives a signal $$\mathsf y$$ which is a mixture of both the desired signal $$\mathsf {x}_1$$ and the interfering signal $$\mathsf {x}_2$$, with some additive noise introduced by the receiver circuits. The goal is to design a receiver processing that can figure out the value of the desired signal $$\mathsf {x}_1$$. 

Mathematically, the received signal can be written as 

$$\mathsf{y} = h_1 \cdot \mathsf{x}_1 + h_2 \cdot \mathsf{x}_2 + \mathsf {w}$$

where 

1. Because all the symbols are transmitted at a carrier frequency, it is conventional to think all the variables as complex valued. Here, we have assumed the standard approach to isolate a single group of transmitted and received symbols. Thus, all the variables are complex scalars in $$\mathcal C$$.
2. $$h_1, h_2$$ are called the fading coefficients for the two corresponding channels. These are usually random, depending on the position of the transmitter and the receiver and the environment around. If the transmitter moves, these fading coefficient can change over time too. However, a wireless system usually has a separate procedure to estimate these coefficients. So, these coefficients are considered as known side information at the receiver, called the **channel state information (CSI)**. 
3. $$\mathsf w$$ is complex Gaussian distributed, independent of all the other variables. This is usually referred to as the "additive Gaussian noise". If there is no interference, the received signal can be described with a conditional distribution of $$\mathsf y$$ given $$\mathsf {x}_1$$, which is Gaussian distributed with mean of $$h_1 x_1$$. The decision of what symbol $$\mathsf{x}_1$$ is transmitted is now a standard hypothesis testing problem, with Gaussian noise, for which the [MAP decision](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) is simple and the optimal solution.




### Why is this difficult?

### Why is this a common problem?

## A solution using nested H-score networks

### $$\blacktriangle$$ Demo: Pytorch implemntation

## The Lessons

### What neural networks to use?

### What metrics to use? 

### The separation: feature learning and assembling

## Going Forward