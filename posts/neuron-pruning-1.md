+++
title = 'Taylor-Series based pruning criteria for weight pruning in neural networks'
date = 2024-07-22
math = 'katex'
author = 'Frederic Mrozinski'
+++

# Summary

This article discusses how the Taylor series motivates two commonly used pruning criteria for neural networks, namely

- Magnitude based pruning
- Fisher information based pruning

The motivation for this blogpost is to highlight that those pruning criteria are closely related while having different bounds on the resulting errors after pruning. 
I created this blogpost because I found this information lacking in many papers that used the concepts introduced below.

# Introduction

A common *fully*-connected neural network - as the name suggests - possesses "synapses", i.e. weights, between *all* neurons of two adjacent layers.
While those weights carry the "knowledge" of such a model, more weights does not always mean, higher "intelligence".
It was in fact already early recognized, that a fully-connected model is generally overparameterized and today, even "small" language models such as BERT function almost equally well with 90% of the weights removed.
Such observations have been summarized by the *Lottery Ticket Hypothesis* saying,

>*A trained fully-connected neural network contains a "much smaller" subnetwork that when reinitialized and trained exhibits at least the same classification performance.*

Or in simpler terms: It is not necessary to keep all synapses between all neurons after training.

These observations motivate *weight pruning*, i.e. "removing" certain weights between certain neurons with the hope of accelerating a model's training and inference time.

# Weight Pruning

Weight pruning is the process of 

1. Identifying unimportant weights between neurons and,
2. "Removing" them by setting their corresponding weight values to 0.

But why? Pruning weights results in weight matrices becoming sparse, which, theoretically,
require fewer multiply-accumulate-operations (i.e. computing time) when being part of a tensor product.
Note that sparsity under loose constraints does not generally translate into less computation time on all hardware, which we do not further cover, here.

And how? While step 2. is trivial, step 1. is not and is the reason for myriads of research papers in the field. In the following, we focus on this step by
presenting two possible and common ways of identifying unimportant weights and motivating them mathematically.

# Taylor based saliency criteria

While we used the term of a weight's *importance* above, the correct jargon would be a weight's *saliency*.
We stick with *importance* for ease of reading.

In the following, we represent a neural network's loss function by its Taylor approximation. If you are not familiar with it, you can still understand all of the presented weight importance measures and just skip the parts that use the Taylor approximation.

Let a classifier neural network $f_{\omega}: \mathcal X \rightarrow [0,1]^c$ be given, where $\mathcal X$ denotes its input space (e.g. set of images) and $c$ the amount of classes to be classified (e.g. number of animals to be differentiated in the images). Let further $L$ denote the network's loss function (we don't formalize ground truth labels, here) and we further implicitly assume $\mathcal X$ to be associated with a probability space with random variable $X$. 
Lastly, let $\omega$ denote the network's weights in a set of possible weights $\Omega$. Then, the network's typical training objective is

{{< rawhtml >}}
$$ \text{min}_{\omega \in \Omega} \space \mathbb{E}_X \left [ L\left (f_{\omega}(X)\right )\right ]. $$
{{< /rawhtml >}}

Further, we will write $f$ instead of $f_{\omega}$ for ease of notation.

Let $\omega_i$ be some weight whose importance / saliency we seek to estimate. Further, in the converged / trained network, let $\omega_i = w$. Then, we can approximate the model's loss function $L$ for $\omega_i=0$ by:

$$L(f(X) \space| \space\omega_i = 0) $$

$$ = L(f(X) \space|\space \omega_i = w) + \mathcal O(|w|)$$

$$ = L(f(X) \space|\space \omega_i = w) - w \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) + \mathcal O(|w|^2)$$

$$= L(f(X) \space|\space \omega_i = w) - w \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) + \frac{w^2}{2} \space \frac{\partial^2}{\partial \omega_i^2}L(f(X) \space|\space \omega_i = w) + \mathcal O(|w|^3).$$ 

Or as we want to minimize the change in loss after pruning a certain weight, we write

$$ E := \mathbb{E}_X\left [ L(f(X) \space|\space \omega_i = 0) - L(f(X) \space| \space\omega_i = w) \right ] $$

$$ = \mathcal{O(|w|)}$$

$$ = -w \space \mathbb{E}_X \left [ \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right ] + \mathcal O(|w|^2) $$

$$ = -w \space \mathbb{E}_X \left [ \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right ] + \frac{w^2}{2}\space \mathbb{E}_X \left [ \frac{\partial^2}{\partial \omega_i^2}L(f(X) \space|\space \omega_i = w) \right ] + \mathcal O(|w|^3)$$ 

But why? The above Taylor approximation enables us to approximate the model's loss with $\omega_i$ pruned *without* actually having to compute the loss in the pruned network.
But don't we have to compute other terms instead, then? Yes - but they are almost trivial to compute as we will show next.

## Maginitude based pruning

The idea of magnitude based pruning lies in *smaller weights (in absolute value) are less important*.

This criterion is very simple but surprisingly effective. E.g. as the authors of the Lottery Ticket Hypothesis have shown, this criterion is effective enough to prune about 90% of weights in common architectures (iteratively).

The intuition behind why it effectively serves as an importance measure is that we would typically expect small changes in a certain weight to lead to small changes in the model's classification performance.
Setting weights to zero which are already close to zero may thus bear little impact on the model's loss.

More formally, the Taylor approximation of

$$ E = \mathbb{E}_X\left [ L(f(X) \space|\space \omega_i = w) - L(f(X) \space| \space\omega_i = 0) \right ] = \mathcal{O}(|w|)$$

bounds the error $E$ by $\mathcal O(|w|)$ which is exactly the motivation for this criterion.

In fact, the error can even be bounded more tightly as 

$$E = -w \space \mathbb{E}_X \left [ \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right ] + \mathcal O(|w|^2),$$

whose term $\mathbb{E}_X \left [ \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right ]$ equals $0$ if the model has fully converged. 

## Fisher-information based pruning

The Fisher-information is a common concept from statistics. Informally, it quantifies how strongly we expect the likelihood function $p$ of a model with parameter $\theta$ to vary as the model's observations $X$ vary according to the distribution with parameter value $\theta = \hat{\theta}$ for some fixed $\hat{\theta}$.
In other words: how much information do different observed values of $X$ carry about the unknown parameter $\theta$ if its true value were $\hat{\theta}$. Formally, the Fisher-information $\mathcal I(\hat{\theta})$ is given by: 

{{< rawhtml >}}
$$ \mathcal I(\hat{\theta}) = \mathbb{E}_{X | \theta = \hat{\theta}} \left [ \left ( \frac{\partial}{\partial \theta} \log p(X | \theta) \space | \space {\theta=\hat{\theta}}\right )^2\right ].$$
{{</ rawhtml >}}

The key-takeaway is that the above formulation only relies on *first* derivatives.

In the context of our model $f$ (which models a density/likelihood function), the parameter $\theta$ is $\omega_i$, $\hat{\theta} = w$, and $X$ is our input data. Then, the Fisher-information of $w_i$ captures how much information the input data carries about $\omega_i$, or how important $\omega_i$ is for the model's classification performance.
It therefore seems promising to use the Fisher-information as a proxy for a weight's importance, right?
 Almost! With some slight modifications...


One fascinating property of the Fisher-information is that (under some regularity conditions), it can also be expressed using the *second* derivative by:

{{< rawhtml >}}
$$ \mathcal I(\hat{\theta}) = \mathbb{E}_{X | \theta = \hat{\theta}} \left [  \frac{\partial^2}{\partial \theta^2} \log p(X | \theta) \space | \space {\theta=\hat{\theta}}\right ]. $$
{{</ rawhtml >}}

The two presented formulations connect the *first* and *second* derivative of the the log-likelihood $\log p$ and show that they even *equal* in expectation.
This powerful property enables us to save tremendous computational effort as computing second derivatives is computationally *very* expensive.

We can make us of this advantage in classifiers that have been trained using the cross-entropy loss: the cross-entropy loss in combination with one-hot targets "collapses" to the logarithm of the predicted probability for the correct class label. Therefore,

$$ \mathbb{E}_X \left [ \frac{\partial^2}{\partial \omega_i^2}L(f(X) \space|\space \omega_i = w) \right ] $$ 

is just the Fisher information of the distribution that the model models for the correct class of each input. Therefore, we rewrite

$$E  = -w \space \mathbb{E}_X \left [ \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right ] + \frac{w^2}{2}\space \mathbb{E}_X \left [ \frac{\partial^2}{\partial \omega_i^2}L(f(X) \space|\space \omega_i = w) \right ] + \mathcal O(|w|^3)$$ 

$$ = -w \space \mathbb{E}_X \left [ \space \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right ] + \frac{w^2}{2}\space \mathbb{E}_X \left [ \left ( \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right )^2 \right ] + \mathcal O(|w|^3),$$

and got rid of second derivatives.

If we assume that the model has been trained to convergence, i.e. that the loss is at a local minimum in expectation, the above further reduces to

$$ = \frac{w^2}{2}\space \mathbb{E}_X \left [ \left ( \frac{\partial}{\partial \omega_i}L(f(X) \space|\space \omega_i = w) \right )^2 \right ] + \mathcal O(|w|^3),$$

because the first term is zero at the local minimum.

What have we gained? Again, we can approximate the weight's importance score by only relying on already computed gradients. The expectation would simply be replaced by the empirical mean over
a sufficiently large data-batch.

# Comparing the above criteria

### Similarities

Both of the above saliency criteria can be motivated by minimizing the resulting pruning error from the Taylor-expansion point of view. They also have in common that they require 
very little compute resources assuming precomputed gradients (which is given in realistic scenarios). They are both easy and quick to implement.

### Differences

The difference between the above criteria lies in the order at which they approximate the resulting pruning error. While magnitude based pruning is guaranteed to quantify the error at $\mathcal O(|w|)$ (and even $\mathcal O(|w|^2)$ for converged models), Fisher-information pruning bounds the error even tighter at $\mathcal{O}(|w|^3)$. 

### In practice

I hope that I can soon, if time permits, run a comparison between both approaches on a common model. I will add that here.


