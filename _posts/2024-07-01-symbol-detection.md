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
> In our previous developments, we introduced the [H-score network](https://lizhongzheng.github.io/blog/2024/H-Score/), which is a way to learn informative features from datasets, similar to a normal neural network, but was claimed to be more flexible. In this post, we use a concrete example in wireless communications to demonstrate this flexibility. In a nutshell, the flexibility comes from changing the objective from learning to solve a specific inference task to learning feature functions that carry information that can be reused and combined with other sources of information. We demonstrate that this approach can help us to design neural-network-based solutions not to a single case, but a parameterized set of cases, as required by this specific engineering problem. We use this example to discuss the general methodology of integrating learning modules to engineering solutions. 


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

The main points of this sequence of works is to introduce the key concepts and building blocks of **feature-centric learning**, where we view feature functions as the basic carrier of information and turn the learning objective from making a certain decision to finding feature functions that carry specific information. This general method makes the learning procedure more controllable and the learning results reusable. The H-score can be viewed as a quality metric for features. The nested H-score network can be viewed as a numerical method for the basic projection operation of features. Both are needed for further development. 


## A Wireless Communication Problem

|![test image](/assets/img/interferenceChannel.png){: width="350" }|
|<b> A Wireless Fading Interference Channel </b>|

<br>

The problem we are working on is called "symbol detection in fading interference channel," which is a classic problem in wireless communications. For a more complete reference [here](https://stanford.edu/~dntse/wireless_book.html) is a wonderful textbook. Roughly the idea is depicted in the figure: a transmitter tries to send a symbol $$\mathsf {x}_1$$ to the receiver, but there is another transmitter, the interferer, which is transmitting his signal at the same time. Thus, the receiver receives a signal $$\mathsf y$$ which is a mixture of both the desired signal $$\mathsf {x}_1$$ and the interfering signal $$\mathsf {x}_2$$, with some additive noise introduced by the receiver circuits. The goal is to design a receiver processing that can figure out the value of the desired signal $$\mathsf {x}_1$$. 

Mathematically, the received signal can be written as 

$$\mathsf{y} = h_1 \cdot \mathsf{x}_1 + h_2 \cdot \mathsf{x}_2 + \mathsf {w}$$

where 

1. Because all the symbols are transmitted at a carrier frequency, it is conventional to think of all the variables as complex-valued. Here, we have assumed the standard approach to isolate a single group of transmitted and received symbols. Thus, all the variables are complex scalars in $$\mathcal C$$.
2. $$h_1, h_2$$ are called the fading coefficients for the two corresponding channels. These are usually random, depending on the position of the transmitter and the receiver and the environment around them. If the transmitter moves, these fading coefficients can change over time, too. However, a wireless system usually has a separate procedure to estimate these coefficients. So, these coefficients are considered as known side information at the receiver, called the **channel state information (CSI)**. 
3. $$\mathsf w$$ is complex Gaussian distributed, independent of all the other variables. This is usually referred to as the "additive Gaussian noise". If there is no interference, the received signal can be described with a conditional distribution of $$\mathsf y$$ given $$\mathsf {x}_1$$, which is Gaussian distributed with a mean of $$h_1 x_1$$. The decision of what symbol $$\mathsf{x}_1$$ is transmitted is now a standard hypothesis testing problem, with Gaussian noise, for which the [MAP decision](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) is simple and the optimal solution.
4. What makes these problems interesting is that the transmitted signals $$\mathsf{x}_1$$ and the signal transmitted by the interferer $$\mathsf{x}_2$$ are both digital signals, in the sense that they are chosen from a discrete set of possible values on the complex plane, which is called a constellation. The most commonly used one is called the [QAM](https://en.wikipedia.org/wiki/Quadrature_amplitude_modulation). 

|![test image](/assets/img/Rectangular_constellation_for_QAM.svg.png){: width="200" }|
|<b> Michel Bakni, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons </b>|

<br>

In this post, we will assume that $$\mathsf{x}_1$$ is equally likely chosen from a binary alphabet $$\mathsf{x}_1 \in \{+1, -1\}$$; and the interference symbol $$\mathsf{x}_2$$ is chosen uniformly from a 16-QAM constellation, which, as shown in the figure, are the 16-point regular grid around the origin. 

This is obviously a special and simple case, but it is sufficient to make our points. We are interested only in detecting the value of $$\mathsf{x}_1$$, which is a binary decision. However, because of the interference signal, our received signal $$\mathsf {y}$$ is corrupted with non-Gaussian noise. 

To see how this works, consider an example shown in the following figure:

|![test image](/assets/img/SymbolDetectionPlots/decision1.png){: width="200" }|
|<b> A "good" case of interference </b>|

<br>

Here, we use the color "red" for $$\mathsf{x}_1 = +1$$, and "blue" for $$\mathsf{x}_1=-1$$. When we transmit $$\mathsf{x}_1= +1$$, the interferer transmits a randomly independently chosen point in his 16-QAM constellation. This causes the noiseless signal $$h_1 \mathsf{x}_1 + h_2 \mathsf{x}_2$$ to be one of the $$16$$ red dots in the figure. If we transmit $$\mathsf{x}_1 = -1$$, this signal before adding noise is one of the $$16$$ blue dots. The additive noise $$\mathsf {w}$$ is then added to the chosen dot, which makes the received symbol $$\mathsf{y}$$ to be somewhere around. 

Our goal is to observe the received symbol $$\mathsf{y}$$ and decide whether it was a red or a blue dot transmitted. Mathematically, the decision is between two **mixed Gaussian distributions**. The case shown above is considered a "good" case with weak interference. That is, even with the interference, the red and the blue cases are rather separable. As long as the additive noise $$\mathsf {w}$$ is not too large, one can simply draw a line to separate the red and the blue cases rather well. In fact, most of the current commercial wireless communication systems use linear decision functions. That is, a straight line is drawn in this figure to separate the red and the blue. 

Unfortunately, the configuration of these "dots" is determined by the fading coefficients $$h_1, h_2$$, which we can observe but cannot control. There is a small probability that the fading can take values that cause an "undesirable" situation. A few of such cases are shown below. If the interference is strong, the two groups of dots can be "interleaved," making it rather difficult to separate them. Furthermore, depending on the value of the fading coefficients, these cases with strong interference can be rather different, as shown in the figure. That is, we cannot design a decision-maker for one case and use it for another case. 

|![test image](/assets/img/SymbolDetectionPlots/decision21.png){: width="300" style="float:left; padding-right:50px"}|![test image](/assets/img/SymbolDetectionPlots/decision31.png){: width="300" style="float:left; padding-right:50px"}|![test image](/assets/img/SymbolDetectionPlots/decision41.png){: width="300" style="float:left; padding-right:50px"}|
|<b> A "marginal" case of interference </b>|<b> A case with strong interference </b>|<b> A different case with strong interference </b>|

<br>


### Why is this difficult?

Analytically, the optimal decision using the likelihood ratio test (LRT) is not difficult to find. The likelihood function of both the red and the blue cases are simply mixture Gaussian densities: the average of $$16$$ equally weighted Gaussian densities. The log-likelihood ratio function can be rather non-linear. Examples are shown in the figure, respectively, corresponding to the $$3$$ cases above. The color code represents the scalar value of this function: red for positive values, which means $$\mathsf{x}_1=+1$$ is more likely, and blue for $$\mathsf{x}_1=-1$$. A simple threshold test can cut out the decision regions. 

|![test image](/assets/img/SymbolDetectionPlots/decision23.png){: width="300" style="float:left; padding-right:50px"}|![test image](/assets/img/SymbolDetectionPlots/decision33.png){: width="300" style="float:left; padding-right:50px"}|![test image](/assets/img/SymbolDetectionPlots/decision43.png){: width="300" style="float:left; padding-right:50px"}|
|<b> Case 1 </b>|<b> Case 2 </b>|<b> Case 3 </b>|


<br>

In practice, however, this analytically optimal decision is rarely implemented for some practical reasons. One of them is the **lack of non-linear processing circuits** in conventional wireless communication devices. Neural networks, as a general-purpose non-linear processor, are now widely used in handheld wireless devices. This makes neural-network-based solutions to this problem an attractive possibility.

The immediate difficulty in building a decision-maker with a neural network is the **lack of training samples**. In this application, we can have both *off-line training* and *on-line training*. Offline training samples can be simulated in the wireless environment based on the industry's extensive channel measurement results. The problem is that there are infinitely many possible situations, as the fading coefficients $$h_1, h_2$$ in our example, are continuous-valued. Thus, whichever specific environment or collection of environments we simulate, the online situation is inevitably different. Online training, on the other hand, is more targeted to the case of interest but is much more expensive. Every symbol we use for training is a symbol we cannot use to carry the communication payload. The wireless channel also changes over time quite fast as the mobile users move, making the online training results expire. 

Another difficulty in this problem is the **high accuracy requirement**. Symbol detections in wireless communication systems require the error rate to be around $$10^{-3}$$. Situations like cases 2 and case 3 in both figures are said to have *strong interference*, which occurs with a probability of the order of $$10^{-2}$$. This means a good receiver needs to do well in all the nice cases like case 1, but what really defines a good receiver is how it handles the strong interference cases. As a result, using the average performance as the objective to train the neural network is not a good idea: how a receiver acts in difficult cases may not count much in the average performance and do not have enough samples in a reasonable-sized training set. 

### Why is this a common problem?

The reason that we like this symbol detection problem is because it is a very typical case for using neural networks in engineering problems. There is clearly a potential of remarkable performance gain, which can come from 

1. the use of neural networks in the place of specialized non-linear circuits; 
2. the training which allows the system to adapt to specific environments. 

The difficulties are also clear:

1. the situations where we want to use the neural networks are almost always different from those we have in the training set; 
2. what we hope to get from the neural networks is not just the optimal solution in one specific case, but a collection of optimal solutions, controlled by parameters (in this example, the channel coefficients.)
3. we also have knowledge of specific structures of the data, (for example the repetitive patterns in the above plots,) which we hope to "tell" the neural network so it does not require more samples to learn this known fact. 

At a higher level, the issue is that neural networks are often operated as blackboxes. In the engineering problems, there is often the need to *inject* the external domain knowledge into the neural networks, in order to control the learning procedure, impose restrictions on learning results, reuse learning results as environment changes, and reduce the overall learning costs. 

The goal of this page is to develop a general solution to this family of problems. 

## A solution using nested H-score networks

### $$\blacktriangle$$ Demo: Pytorch implemntation

## The Lessons

### What neural networks to use?

### What metrics to use? 

### The separation: feature learning and assembling

## Going Forward
