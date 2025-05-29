---
layout: post
title: "MSE Bias-Variance Decomposition: A Gentle Walkthrough"
date: 2025-05-29 12:00:00 +0000
---

## Contents

- [Preamble](#preamble)
- [Why Bother?](#why-bother)
- [Assumptions and Definition](#assumptions-and-definition)
- [Derivation](#derivation)
- [Interpretation](#interpretation)
- [Practical Implications](#practical-implications)
- [Concluding Thoughts](#concluding-thoughts)


## Preamble 

It's pretty normal to see a discussion of some sort on the bias-variance decomposition of the mean squared error (of a model) in most machine learning textbooks, often within a larger chapter or sub-section on the bias-variance tradeoff. In almost all of these textbooks, however, getting through the derivation of the expression feels a lot like a tricky obstacle course (think Wipeout or Ninja Warrior), making the entire process seem arbitrarily hard and even somewhat pointless.

My main motivation for writing this post was to make the derivation more accessible, even if only slightly, by clarifying what the underlying assumptions are, and ensuring that any elements of the notation which I'd deem crucial to include (e.g., the subscript of the expectation function; you'd be surprised at how much clearer leaving this in makes the entire process) **are** included. You'd benefit the most from this post if you are familiar with the expectation function and its properties, as well as some very basic machine learning concepts (most of the examples in this post centre around linear regression, but decision trees do make a very brief appearance a little towards the end).

I should also preface this by saying that this is intended to be a moderately informal, tutorial-esque post, and hence my tone will remain conversational for the most part. I plan on doing slightly more research-focused posts in the future, but for my first one I thought I'd start with something relatively "light". Tags will be featured to indicate which posts are more tutorial-style, and which ones are more centred around research. 

## Why Bother?

The essence of bias-variance decomposition is to mathematically formalise how well our learning algorithm is expected to perform on average **for a given problem type** (i.e., across infinitely many different potential datasets for the problem), not on a single, specific dataset within that problem type. Moreover, it lets us identify why a regression model might yield a high error, and consequently allow us to tweak relevant aspects of our learning algorithm to reduce it. 

Recall that, in practice, we normally divide the sum of the errors squared by the sample size $n$ to obtain the mean squared error (as a loss). With that in mind, let's say we train a linear regression model on the well known Boston Housing dataset:

- We could then take this model and calculate its MSE **on the same dataset**, and make a conclusion about the model's efficacy that way. In this case, we've calculated the model's ***training error***, which tells you how well **that particular model** fits **on the data it was trained on**. The training error serves as a very optimistic measure of how well the model might perform on new, unseen data, but it is insufficient as a measure on its own. 

- Alternatively, we could calculate its MSE on a separate "test" dataset – let's say "Nick's Boston Housing Dataset". Doing so tells you how well your particular model has performed on a particular instance of new, unseen data, but tells you only a little about how well your model might perform on **any** new, unseen data in general (and even less about the general efficacy of your learning algorithm). Nonetheless, it's worth noting that the MSE we obtained from the test set is still a much better estimate of this more general error, which we fittingly call the ***generalisation error***, than the training error was. 

We've therefore established that we don't have a system in place to measure how well we expect our learning algorithm to perform on average when it encounters new data. Evaluating the MSE on a test set might estimate this generalisation error better than the training error, but it is still only an estimate, since it considers only a single, finite instance of new data. 

## Assumptions and Definition

Some formal assumptions first. 

Let's assume our dataset $D$ is a collection of input-output (or feature(s)-target) pairs $(x_i, y_i)$, where each pair is i.i.d from some true data-generating distribution $P(X,Y)$. In other words:

$$
\begin{split}
(x_i, y_i) &\sim P(X,Y)  \\ \\ 
D &= \set{(x_i, y_i)}^n_{i=1} = \set{(x_1,y_1), (x_2, y_2),\dots, (x_n, y_n)} 
\end{split}
$$

Each output $y_i$ is comprised of the true regression function, $f(x_i)$, and a specific noise term $\epsilon_i$. 
It's worth noting that all the noise terms are i.i.d with a mean of 0 and variance $\sigma^2$. i.e.:

$$
y_i = f(x_i) + \epsilon_i
$$

Let's also say that our selected learning algorithm $\mathcal{A}$ (e.g. linear regression) takes a dataset $D$ and outputs a model $\hat{f}_D = \mathcal{A}(D)$. For example, a linear regression model trained on the Boston Housing Dataset would yield $\hat{f}_D = \text{Linear Regression}(\text{Boston Housing})$. 

Let's also define a new, unseen datapoint $(x_0, y_0)$. This datapoint, which I will sometimes refer to as the test point, **does** follow the same true underlying distribution $P(X,Y)$ of all the datapoints in the training dataset $D$, but it is imperative to remember that **it is not part of the training dataset $D$**. It is an entirely new datapoint. Furthermore, for this derivation, $x_0$ is considered fixed. Whilst $x_0$ itself might be drawn from the marginal distribution $P(X)$, for the purpose of analyzing the error at this specific point, we treat $x_0$ as a non-random quantity that we've selected. Averaging over all possible $x_0$'s therefore corresponds to infinitely picking new test points. In other words:

$$
(x_0, y_0) \sim P(X,Y) \text{ and } (x_0, y_0 ) \notin D 
$$

We therefore define a general mean squared error for a particular test input as follows:

$$
\text{MSE}(x_0) = \mathbb{E}_{\epsilon_0, D}[(y_0 -\hat{f}_D(x_0))^2]
$$

Before we start going through the math, there's a couple of things I should make explicit just to make the intuition behind what we're doing a bit clearer. 

The subscript of the expectation function, i.e., $\mathbb{E}_{\text{this part}}$, shows you the distribution that you're averaging over (i.e., what to substitute for a probability mass/density function). Here, we're finding the expectation of the squared error with respect to a joint probability distribution of two random variables: $\epsilon_0$ and $D$. In other words, we are finding the expected MSE of our learning algorithm on average across all possible datasets $D$ and noise $\epsilon_0$ at the test point.

$D$ and $\epsilon_0$ being random variables should, hopefully, make sense. Recall that $D$ is made up of input-output pairs $(x_i, y_i)$, where $x_i$ and $y_i$ themselves are random variables, drawn from a true data-generating distribution $P(X,Y)$, and thus by extension $D$ itself is a random variable. 

I've specified $\epsilon_0$ as the other random variable (instead of $y_0$ or $(x_0, y_0)$ for example) because it is technically the only random component in our test point. Recall that $y_0 = f(x_0) + \epsilon_0$, where $f$ is the true regression function (inaccessible to us) and $\epsilon_0$ is the noise term. As described earlier, we assume $x_0$ is fixed and arbitrarily selected (and done infinitely many times), and thus $f(x_0)$ is by extension also not random, since the true regression function $f$ remains fixed for all the datapoints in our dataset. Consequently, the randomness in a given test point stems only from its noise term $\epsilon_0$, and thus for conciseness I've used $\epsilon_0$ instead of $(x_0, y_0)$ in the expectation subscript. 


Hopefully that makes sense. Back to the math!

## Derivation

Let's expand:

$$
\mathbb{E}_{\epsilon_0, D}[(y_0 -\hat{f}_D(x_0))^2] = \mathbb{E}_{\epsilon_0, D}[(f(x_0) + \epsilon_0 -\hat{f}_D(x_0))^2] 
$$

And here's where we apply what I like to call the Add-Subtract trick. In this case, the term we add-subtract is $\mathbb{E}_{D}[\hat{f}_D(x_0)]$, and we can do this because $\mathbb{E}_D[\hat{f}_D(x_0)]$ is a theoretical constant (the reason why will be explained a little later):

$$
\mathbb{E}_{\epsilon_0, D}[(f(x_0) +\epsilon_0-\hat{f}_D(x_0))^2] = \mathbb{E}_{\epsilon_0, D}[((f(x_0) - \mathbb{E}_{D}[\hat{f}_D(x_0)]) + (\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0)) + \epsilon_0)^2]
$$

Okay, quite a few terms. Let's let:

$$
\begin{split}
A &=  f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)] \\ \\
B & = \mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0) \\ \\ 
C &= \epsilon_0
\end{split}
$$

Therefore let's re-write the expectation using our variables:

$$
\begin{split}
\mathbb{E}_{\epsilon_0, D}[((f(x_0) - \mathbb{E}_{D}[\hat{f}_D(x_0)]) + (\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0)) + \epsilon_0)^2] &= \mathbb{E}_{\epsilon_0, D}[(A + B + C)^2] \\ \\ & =  \mathbb{E}_{\epsilon_0, D}[A^2+B^2+C^2 + 2(AB+AC+BC)] \\ \\ & = \mathbb{E}_{\epsilon_0, D}[A^2] + \mathbb{E}_{\epsilon_0, D}[B^2] + \mathbb{E}_{\epsilon_0, D}[C^2] +2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC] 
\end{split}
$$

The expansion $\mathbb{E}[X + Y + Z] = \mathbb{E}[X] + \mathbb{E}[Y] + \mathbb{E}[Z]$ is allowed due to the linearity of the expectation function. Now that we've done that, let's consider each individual term, and then add them together to get the final expression, starting with the first:

$$
\mathbb{E}_{\epsilon_0, D}  [A^2] =   \mathbb{E}_{\epsilon_0, D}[(f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2]
$$

Hold on, let's think for a second:

1) Neither $f(x_0)$ nor $\mathbb{E}_{D}[\hat{f}_D(x_0)]$ are dependent on $\epsilon_0$, since $f(x_0)$ is the true (theoretical) regression function explicitly excluding the noise (remember: we only really introduce $\epsilon_0$ in the definition of $y_0= f(x_0) + \epsilon_0$, which shows that $f(x_0)$ is not dependent on $\epsilon_0$), and $\mathbb{E}_D[\hat{f}_D(x_0)]$ evidently has no link to $\epsilon_0$. 

2) Neither $f(x_0)$ nor $\mathbb{E}_D[\hat{f}_D(x_0)]$ are dependent on the $D$ in the outer expectation as well. The dataset $D$ we train our model on has no effect on the true regression function $f$, consequently leading to $D$ and $f(x_0)$ being independent. $\mathbb{E}_D[\hat{f}_D(x_0)]$ is a bit more interesting to think about. At first glance, you might think that it **should** depend on the dataset $D$, since the dataset selected defines what $\hat{f}_D$ becomes. This reasoning is absolutely valid: $\hat{f}_D(x_0)$ does vary with $D$. However, the expression $\mathbb{E}_D[\hat{f}_D(x_0)]$ denotes an expectation over all possible datasets; an "average" prediction of our learning algorithm having considered infinitely many datasets $D$. Doing this makes the term deterministic; it doesn't matter what our $D$ is in the outer expectation term as it won't affect it, since $\mathbb{E}_D[\hat{f}_D(x_0)]$ has already considered an infinite number of datasets $D$. $\hat{f}_D(x_0)$ and $\mathbb{E}_D[\hat{f}_D(x_0)]$ are two different numbers here: one is dataset-specific and is based on the current $D$, whilst one has been "pre-computed" by considering all the possible values $D$ could take, and is therefore a theoretical constant.

Okay, so we've established that $f(x_0)$ and $\mathbb{E}_{D}[\hat{f}_D(x_0)]$ are both independent of both $D$ and $\epsilon_0$. That means taking the expectation of these two with respect to the joint probability distribution $\set{D, \epsilon_0}$ is moderately "pointless", since neither $D$ nor $\epsilon_0$ changing will affect their values. Hence, they're constants (w.r.t. the outer expectation over $D$)! And the expectation (w.r.t. anything) of a constant is just the constant itself. We can therefore just say:

$$
\begin{aligned}
\mathbb{E}_{\epsilon_0, D}  [A^2]  = \mathbb{E}_{\epsilon_0, D}[(f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2]  =  (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2 
\end{aligned}
$$

***First term done***. Next:

$$
\mathbb{E}_{\epsilon_0, D}[B^2] =\mathbb{E}_{\epsilon_0, D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2]
$$

Again, let's think about it.

1) We've already established that $\mathbb{E}_D[\hat{f}_D(x_0)]$ is independent of both $\epsilon_0$ and $D$. 

2) $\hat{f}_D(x_0)$, however, **is dependent** on $D$, but not on $\epsilon_0$. Therefore, whilst we can ignore $\epsilon_0$ in the subscript of the expectation, we can't ignore $D$. 

Therefore:

$$
\mathbb{E}_{\epsilon_0, D}[B^2] =\mathbb{E}_{\epsilon_0, D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2] = \mathbb{E}_{D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2]
$$

***Second term done***. Third term now:

$$
\mathbb{E}_{\epsilon_0, D}[C^2] = \mathbb{E}_{\epsilon_0, D}[\epsilon_0^2] = \mathbb{E}_{\epsilon_0}[\epsilon_0^2]
$$

This should make sense:

1) $\epsilon_0$ is the noise term for the output $y_0$ in the new data point ($x_0$, $y_0$), and earlier we explicitly specified that $(x_0, y_0) \notin D$, which implies $\epsilon_0 \notin D$. Therefore $\epsilon_0$ is independent of $D$, so the subscript of the expectation just becomes $\mathbb{E}_{\epsilon_0}$. 

2) The expected value of any particular instance of noise, i.e., $\epsilon_i$ (such as $\epsilon_0$), with respect to its own distribution, is 0: this makes sense since some noise terms might be above the mean, and some might be below, and since all the instances of noise are equally likely they essentially just “cancel out”. However, since we're squaring it here, all the negatives become positives, and as such the expected value cannot be 0.

***Third term done***. Final term's all that's left:

$$
2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC] 
$$

We can use the linear properties of the expectation function to expand that out a little:

$$
\begin{split}
2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC]   &= 2(\mathbb{E}_{\epsilon_0, D}[AB] + \mathbb{E}_{\epsilon_0, D}[AC] + \mathbb{E}_{\epsilon_0, D}[BC]) 
\end{split}
$$

Some things to remember:

1) We've established that neither of the terms in $A = f(x_0) - \mathbb{E}_D[\hat{f}_D(x_0)]$ depend on $\epsilon_0$ or the dataset $D$, so in this context $A$ is a constant (and thus can be factored out if needed, again due to the linear properties of the expectation function). 

2) The second $\hat{f}_D(x_0)$ in $B = \mathbb{E}_D[\hat{f}_D(x_0)] - \hat{f}_D(x_0)$ depends on the dataset $D$ in the outer expectation, but neither terms depend on $\epsilon_0$, so we can ignore the $\epsilon_0$ in the expectation subscript.

3) $C=\epsilon_0$ is independent of the dataset $D$ but dependent on $\epsilon_0$. 

Let's consider the three terms here separately then, starting with:

$$
\mathbb{E}_{\epsilon_0, D}[AB]
$$

We've said that $A$ can be factored out, and that $B$ is only dependent on the dataset $D$, thus:

$$
\begin{split}
\mathbb{E}_{\epsilon_0, D}[AB] = A \cdot \mathbb{E}_D [B] &= (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)]) \cdot \mathbb{E}_D[\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0)] \\ \\
&= (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)]) \cdot (\mathbb{E}_D[\mathbb{E}_D[\hat{f}_D(x_0)]] - \mathbb{E}_D[\hat{f}_D(x_0)]) \\ \\
& = (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)]) \cdot (\mathbb{E}_D[\hat{f}_D(x_0)] -\mathbb{E}_D[\hat{f}_D(x_0)])  
= (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)]) \cdot 0 = 0 \\ \\ 
\end{split}
$$

Okay. $\mathbb{E}_{\epsilon_0, D}[AB] = 0$. Next (keeping in mind that $C$ only depends on $\epsilon_0$ and that the expected value of any noise term w.r.t. itself is 0, since by definition the noise is centred):

$$
\begin{split}
\mathbb{E}_{\epsilon_0, D}[AC] = A \cdot \mathbb{E}_{\epsilon_0}[C] &= (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)]) \cdot  \mathbb{E}_{\epsilon_0} [\epsilon_0] \\ \\ 
& = (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)]) \cdot 0 = 0
\end{split}
$$

That's neat. For the final one, $\mathbb{E}_{\epsilon_0, D}[BC]$, it's not as easy as just factoring out one of the terms, since neither are constant. However, recall that:

1) $B$ is dependent on $D$, but independent of $\epsilon_0$.

2) $C$ is dependent on $\epsilon_0$, but independent of $D$. 

By extension, that means $B$ and $C$ are independent! This makes life a lot easier for us, as we can apply a particular property of the expectation function and just:

$$
\begin{split}
\mathbb{E}_{\epsilon_0, D}[BC] &= \mathbb{E}_{\epsilon_0, D}[B] \cdot \mathbb{E}_{\epsilon_0, D} [C]  \\ \\ 
& = \mathbb{E}_D[B] \cdot \mathbb{E}_{\epsilon_0}[C] \\ \\ 
& = \mathbb{E}_{D}[\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0)] \cdot \mathbb{E}_{\epsilon_0}[\epsilon_0] \\ \\ 
& = 0 \cdot 0 = 0
\end{split}
$$

The jump from the third to the fourth line is based off intuition we built earlier:

Therefore:

$$
\begin{split}
2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC] &= 2(\mathbb{E}_{\epsilon_0, D}[AB] + \mathbb{E}_{\epsilon_0, D}[AC] + \mathbb{E}_{\epsilon_0, D}[BC])  \\ \\ 
&= 2(0 + 0 + 0) = 0 \\ \\ \therefore 2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC]  &= 0
\end{split}
$$

Just a recap of our four terms in their context:

$$
\begin{split}
\mathbb{E}_{\epsilon_0, D}[((f(x_0) - \mathbb{E}_{D}[\hat{f}_D(x_0)]) + (\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0)) + \epsilon_0)^2] &= \mathbb{E}_{\epsilon_0, D}[A^2] + \mathbb{E}_{\epsilon_0, D}[B^2] + \mathbb{E}_{\epsilon_0, D}[C^2] +2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC] 
\\ \\ 
\mathbb{E}_{\epsilon_0, D}  [A^2] &= ( f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2
\\ \\ 
\mathbb{E}_{\epsilon_0, D}[B^2] &= \mathbb{E}_{D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2]
\\ \\ 
\mathbb{E}_{\epsilon_0, D}[C^2] &= \mathbb{E}_{\epsilon_0}[\epsilon_0^2]
\\ \\ 
2\mathbb{E}_{\epsilon_0, D}[AB + AC + BC]  &= 0
\end{split}
$$

Therefore:

$$
\mathbb{E}_{\epsilon_0, D}[((f(x_0) - \mathbb{E}_{D}[\hat{f}_D(x_0)]) + (\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0)) + \epsilon_0)^2] = ( f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2 + \mathbb{E}_{D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2] +\mathbb{E}_{\epsilon_0}[\epsilon_0^2]
$$

Or, to refer to it in its original context

$$
\text{MSE: }  \mathbb{E}_{\epsilon_0, D}[(y_0 - \hat{f}_D(x_0))^2] = (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2 + \mathbb{E}_{D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2] + \mathbb{E}_{\epsilon_0}[\epsilon_0^2]
$$

## Interpretation

Having decomposed the original expression, we're left with the sum of three terms:

$$
\tag{1}
( f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2
$$

$$
\tag{2}
\mathbb{E}_{D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2]
$$

$$
\tag{3}
\mathbb{E}_{\epsilon_0}[\epsilon_0^2]
$$

$(1)$ here is referred to as the $\text{Bias}^2$ term (or more precisely the $(-\text{Bias})^2$ term, since $\text{Bias}$ is conventionally $\mathbb{E}_D[\hat{f}_D(x_0)] - f(x_0)$). This makes intuitive sense, since the "bias" of our model is essentially how inaccurate, on average, our learning algorithm's prediction is when compared to the **true (theoretical) regression function** $f(x_0)$. Think of $\mathbb{E}_D[\hat{f}_D(x_0)]$ as the prediction of an "average model" produced by our learning algorithm. Bias is therefore how far this "average model" $\mathbb{E}_D[\hat{f}_D(x_0)]$ is from the truth $f(x_0)$. 

<p>
$(2)$ here is the <b>variance</b> of our model. Again, this should make intuitive sense, since variance quantifies the spread of a random variable around its mean. If our model is "flexible" (which often occurs for more complex models, e.g., polynomial linear regression, in contrast to simple linear regression), this variance term will be higher, since our model will "try harder" to adapt to the different noise terms in our specific dataset $D$, meaning a different dataset might completely alter the "shape" of the model entirely (since with a new dataset comes new, potentially drastically different, noise). Thus, on average, for a particular unseen input $x_0$, our complex model $\hat{f}_D(x_0)$ would give a result pretty far from the "average model" $\mathbb{E}_D[\hat{f}_D(x_0)]$. 
</p>

$(3)$ here is our "irreducible error" - often denoted as $\sigma^2$. This term is, by definition, the variance of the noise $\epsilon_0$ at the test point. Notice how this third term is **completely independent** from our dataset $D$; hence "irreducible". Regardless of how good our model is, and regardless of the dataset $D$ we select, the noise that is inherent in essentially all regression problems will persist. This random instance of noise $\epsilon_0$ cannot be directly affected; it is simply "there". 

To summarise, our MSE can be broken down into:

$$
\begin{split}
\mathbb{E}_{\epsilon_0, D}[(y_0 - \hat{f}_D(x_0))^2]  &= (f(x_0)- \mathbb{E}_{D}[\hat{f}_D(x_0)])^2 + \mathbb{E}_{D}[(\mathbb{E}_{D}[\hat{f}_D(x_0)] - \hat{f}_D(x_0))^2] + \mathbb{E}_{\epsilon_0}[\epsilon_0^2]
\\ \\ 
&= \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\end{split}
$$

## Practical Implications

Upon a little deliberation, you might notice that most of that final expression is actually incomputable. Anything that involves an $\mathbb{E}_D$ is immediately inaccessible to us, as we don't know the PMF/PDF of the true underlying distribution for our dataset, which means that we can't compute either the variance or the $\mathbb{E}_D[\hat{f}_D(x_0)]$ in the bias term. The same applies for the true regression function $f(x)$ – we can never achieve a perfect model such that $f(x_0) - \hat{f}_D(x_0) = 0$; only a very close estimate. I've also already discussed our inability to control the irreducible error $\sigma^2$.  

So what's the point? Why bother breaking down the MSE to these three terms, if we can't compute them anyway?

The bias-variance decomposition we've just done serves as a framework for understanding the broader bias-variance tradeoff.

To reiterate: in practice, we calculate the total Mean Squared Error (MSE) by using a validation or test set, and divide the sum of the errors squared by the sample size $n$ to obtain the mean squared error. It is natural to then ask: how does this help justify why we've decomposed it? If we know how to calculate it, why bother understanding that it's composed of a $\text{Bias}^2$ term, a $\text{Variance}$ term, and an $\text{Irreducible Error}$ term?    

I asked myself this question, and found the answer using a simple analogy:

Let's say I was interested in predicting the housing prices in Boston given details/features of particular houses. In doing so, I have trained two models on a single dataset:

- Model 1: A linear regression model. A very simple one, containing 3 linear variables; one for the number of bedrooms, one for its square footage, and another for the distance to the city centre. 
- Model 2: A very deep, unpruned decision tree, which considers all the possible features a house might have (even the irrelevant ones), and consequently has very many leaf nodes.

I evaluate an estimate for the MSE of both using a test set (i.e., data it hasn't seen before), and get the same very high number for both. I am therefore confused by this result, as they are completely different models, with the second being far more complex than the first.

Without any knowledge of what a model's "bias" or "variance" might refer to, I would probably struggle to answer this question. The usefulness of the decomposition of MSE is most apparent here, since I now have three very clear possible sources of a high MSE:  

### Bias

Recall that bias measures how far, on average, our learning algorithm's prediction deviates from the true regression function value $f(x_0)$. A high bias means the model is systematically wrong, often because it is too "simple" and has failed to capture complex relationships in the data.

1) Model 1 (Linear Regression): This model most likely had **high bias**. Housing prices fluctuate and depend on complex, often non-linear, relationships between many of a house's individual features. A simple linear regression model containing only three variables almost definitely failed to capture such relationships, meaning, on average, its predictions deviate significantly from the value of the true regression function $f(x_0)$. It is **underfitting** the data. 

2) Model 2 (Deep Decision Tree): This model most likely had **low**, or at the very least **lower**, bias. A deep, unpruned decision tree is extremely "flexible", and can create a very large number of splits based on many features, consequently allowing it to approximate highly complex and non-linear relationships. Because it can represent highly complex functions, if we were to average its predictions over an idealised, infinite set of training datasets, this average could, if it were "rich"/"flexible" enough, approach the true underlying regression function $f(x)$, since there are enough patterns present to fully capture all the complex relationships. The problem is, we're not training it on an infinite number of datasets, but rather on one. A low training error on one dataset is not a guarantee that the model will perform well on all other datasets. Reasoning behind this will be provided below, when considering the inherent high variance in many complex models.

### Variance

<p>This term measures how much our model's predictions for a specific point $x_0$ tend to fluctuate if we were to train it on many different training datasets $D$. Specifically, it measures the expected squared deviation of what our model predicts (from a specific dataset) $\hat{f}_{D}(x_0)$ from the average value it might predict across all possible datasets $\mathbb{E}_{D}[\hat{f}_D(x_0)]$: </p>

<p>
1) Model 1 (Linear Regression): This model almost definitely had <b>low variance</b>. Due to its simple structure, the best-fit line, or hyperplane, produced by the algorithm doesn't change too drastically when trained on different Boston Housing Price datasets. Its linear nature means it quite literally does not have the capacity to adapt (or, alternatively, "bend") to gentle fluctuations in the data (which is often caused by the noise term $\epsilon_i$), so the coefficients for the three linear variables would remain relatively stable, and would all most likely hover around the mean model prediction $\mathbb{E}_{D}[\hat{f}_D(x_0)]$. However, this stability (from low variance) isn't necessarily a good thing if that mean model prediction $\mathbb{E}_D[\hat{f}_D(x_0)]$ is far from the true value $f(x_0)$ – which is exactly what happens with high bias. <br> <br>
2) Model 2 (Deep Decision Tree): This model most likely had <b>high variance</b>. Its nature as a complex model allows it to adapt to even the most subtle fluctuations in the training data's output values. For a deep tree, the choice of the very first split can be influenced by a few data points. If those points are slightly different in another dataset (due to noise), that first split might change, leading to a cascade of different subsequent splits and thus a completely different tree structure. This means our model is very dependent on the dataset it is trained on; a slightly different dataset might yield a completely different model. As a consequence, you are far more likely to get model predictions $\hat{f}_D(x_0)$ which differ greatly from the mean model prediction (across all datasets) $\mathbb{E}_{D}[\hat{f}_D(x_0)]$. This phenomenon is known as <b>overfitting</b>. 
</p>

### Irreducible Error

This is the error component due to inherent randomness or noise in the data-generating process itself which no model, no matter how good, can eliminate. The same dataset is used to train both models, so this simply serves as a baseline level of error we cannot get below. 


To conclude that analogy, yes, a linear regression model and a deep decision tree are two completely different models, so it is only natural to be confused by an identical (high) MSE, even if trained on the same dataset. However, we now know that the reason behind the two MSEs are different, namely:

1) Linear Regression (Underfitting): $\text{High MSE} ≈ (\text{Large Bias})^2 + (\text{Small Variance}) + \text{Irreducible Error}$ 

2) Deep Decision Tree (Overfitting): $\text{High MSE} ≈ (\text{Small Bias})^2 + (\text{Large Variance}) + \text{Irreducible Error}$ 

## Concluding Thoughts 

In short, the bias-variance decomposition of MSE gave us a language to diagnose and address model issues (in this case, specifically in the context of regression problems), and revealed to us how we might reason about model complexity, performance, and generalisation. Crucially, it explains **why** the bias-variance tradeoff exists, and provides a motivation for careful learning algorithm selection. 

In the future, I’ll most probably talk through the practical strategies for navigating the trade-off, as well as extend the notion of a diagnostic framework to classification tasks (as MSE tends to be very specific to regression). A more interesting insight might also be one into the contemporary nuances introduced by applying the concept to deep learning!