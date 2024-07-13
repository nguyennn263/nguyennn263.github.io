---
title: 'Markov chain part II'
date: 2024-07-05
categories: [Knowledge]
tags: [machine_learning, probability, markov_chain, markov_process]
description:  Markov chain part II
math: true
published: true
media_subpath: /assets/markov-chain/
# image:
#     path: alt-img.png
---

## Introduction
To continue this <a href="https://nguyennn263.github.io/posts/markov-chain/" target="_blank">post</a>, we will explore the "stationary distribution" and solve the example about Markov chain.

## Stationary distribution 
Given transition matrix $P$, let $\pi_j = \lim\limits_{n \to \infty}\pi P_{ij}^n$

E.g: Market segmentation problem from the previous post. You can see my <a href="https://nguyennn263.github.io/posts/markov-chain/markov.ipynb" target="_blank">code</a>.

First we need to import numpy and matplotlib.

````python
import numpy as np
import matplotlib.pyplot as plt
````

Declare $\pi^{{0}}$ and $P$ and loop untill the $\pi^{(i)} \approx\pi^{(i+1)}$

````python
pi = np.array([0.2, 0.5, 0.3])
P = np.array(
    [[0.8, 0.1, 0.1],
     [0.07, 0.9, 0.03],
     [0.083, 0.067, 0.85]]
)
pi_n = pi
eps = 1e-3

distributions = [pi]

tmp = pi @ P
while np.all(np.abs(tmp - pi_n)) > eps: 
    pi_n = tmp
    tmp = pi_n @ P
    distributions.append( pi_n)
print(pi_n)
````
````plaintext
[0.27238415 0.45502202 0.27259384]
````

We can plot area char to show distribution for every iteration.

````python
distributions = np.array(distributions)
plt.stackplot(range(len(distributions)), distributions.T, labels=['A', 'B', 'C'], alpha=0.8)
plt.show()
````

![Distributions for every iteration](prob1.png)
_Distributions_

We can see that the distribution will converge to $\pi^{(n)}$.

> **Question:** When do we have a stationary distribution? Is there only one? How do we find it using linear algebra? 
{: .prompt-info }

To answer this, I will let you explore yourself because this series will introduce the brief about Markov chain.

We have the stationary distribution so we can predict that after $n$ months, the market will be stable and store B will have the most number of customers.

## Conclusion
Markov chain is a topic in *Applied Mathematics and Statistics* which is an optional subject in my university. 

Thanks for your reading!  

