---
title: Markov chain 
date: 2024-07-05
categories: [Knowledge]
tags: [machine_learning, probability, markov_chain, markov_process]
description:  Markov chain 
math: true
published: true
media_subpath: /assets/markov-chain/
image:
    path: alt-img.png
---

<!-- ### Nội dung
- [1. Phát biểu bài toán](#-Phat-bieu-bai-toan)
- [2. Định nghĩa xích Markov](#-Dinh-nghia-xich-Markov)
- [3. Ma trận chuyển trạng thái và lược đồ](#-Ma-tran-chuyen-trang-thai-va-luoc-do)
- [4. Phân phối xác xuất](#-Phan-phoi-xac-xuat)
- [5. Minh hoạ 1 quá trình Markov](#-Minh-hoa-1-qua-trinh-Markov)
- [6. Tổng kết](#-Tong-ket) -->

In this post, we will explore Markov chain.

<!-- <a name="-Phat-bieu-bai-toan"></a> -->
## 1. Problem statement.
In an area with 1,000 people, there are 2 stores that sell the same type of product. Initially, the statistics show that 800 people buy the product from store A, 200 people buy from store B.  After a period of competition, advertisement, and promotions to attract customers, the statistics are updated and showed the following situations:
- The probability that a person who previously bought from store A still buys from store A is $p_{11}$
- The probability that a person who previously bought from store A now buys from store B is $p_{12}$
- The probability that a person who previously bought from store B now buys from store A is $p_{21}$
- The probability that a person who previously bought from store B still buys from store B is $p_{22}$

**Question**: After 3 months, 6 months,..., $n$ months later, what will be the proportion of people buying from each store?

![The change in state over time(similar to S1)](phat-bieu.png)
_The change in state over time(similar to S1)_

![Illustrative diagram](minh-hoa.png)
_Illustrative diagram_

<!-- <a name="-Dinh-nghia-xich-Markov"></a> -->
## 2. Definition of Markov chain
### 2.1. State space


For a random process $\set{X_n, n = 0, 1, 2, ...}$, the set $S$ consisting of all possible values of the sequence of random variables $(X_n, n = 0, 1, 2,...)$ which is call **state space** of the random process $(X_n)_n$.

- Here are the examples with their state spaces:
    - Shopping habit problem: $S = \set{0, 1}$
    - Assuming weather conditions can only be rainy or sunny: $S = \set{R(rainy), S(sunny)}$
    - If $X(t)$ represents the number of people queued at an ATM at time $t$, then the state space of the random process ${X_t}$ is: $S = \set{0, 1, 2,...}$

### 2.2. Markov chain
A random process $\set{X_n, n = 0, 1, 2,...}$ and a state space $S \subset \set{0, 1, 2,...}$. A random process is a Markov chain if:
$$
\begin{equation} \notag
P(X_{n+1} = j|X_n = i, X_{n−1} = i_{n−1}, . . . , X_0 = i_0) = P(X_{n+1} = j|X_n = i), \forall n, i, j, i_0, i_1, . . . , i_{n−1}
\end{equation}
$$

**Note:**
- If set $S$ is finite, then we have a finite Markov chain.
- Let $p_{ij} = P(X_{n+1} = j \| X_n = i)$, in this case $p_{ij}$ is independent of $n$:
$$
\begin{equation} \notag
p_{ij} = P(X_1 = j|X_0 = i) = P(X_2 = j|X_1 = i) = P(X_3 = j|X_2 = i) = ...
\end{equation}
$$

<!-- <a name="-Ma-tran-chuyen-trang-thai-va-luoc-do"></a> -->
## 3. Transition matrix and state diagram
If state space is $S = \set{1, 2, 3, . . . ,r}$ and $p_{ij} = P(X_{n+1} = j \| X_n = i)$, then this matrix:

$$
\ {P} = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1r} \\
p_{21} & p_{22} & \cdots & p_{2r} \\
\vdots & \vdots & \ddots & \vdots \\
p_{r1} & p_{r2} & \cdots & p_{rr}
\end{pmatrix}
$$

is called Transition matrix after **one** step.

**Note** 
- We always have $p_{ij} >= 0$ for all $i$ and $\sum_{k = 1}^{r} p_{ik} = \sum_{k = 1}^{r} P(X_{n+1} = k\|X_n = i) = 1$ 

<!-- <a name="-Phan-phoi-xac-xuat"></a> -->
## 4. State probability distributions
### 4.1. Probability of transitioning after n steps
The probability of transitioning after $n$ steps is defined as follows:
$$
\begin{equation} \notag
p_{ij}^{n} = P(X_{n+m} = j|X_m = i) = P(X_n = j|X_0 = i)
\end{equation}
$$

where the convention is
$$p_{ij}^{(0)} = 
\begin{cases} 
0 & \text{if } i \ne j \\ 
1 & \text{if } i = j  \\
\end{cases}
$$
### 4.2. State probability distributions
If the state space is $S = \set{1, 2, 3,... ,r}$, the distribution of the random variable $X_n$ is given by:
$$
\begin{equation} \notag
\pi^{(n)} =  [P(X_n = 1)\quad P(X_n = 2) \quad··· \quad P(X_n = r)]
\end{equation} 
$$

The distribution $\pi^{(0)}$ of the variable $X_0$ is called the initial distribution of the system.
### 4.3. Theorem
For a Markov chain $\set{X_n, n = 0, 1, 2,...}$ with tranposition after 1 step $P$, we have:

$$
\begin{equation}  \notag
\pi^{(n+1)} = pi^{(n+1)}P,  \text{với } n = 0, 1, 2,...
\end{equation} 
$$

$$
\begin{equation} \notag
\pi^{(n)} =  pi^{(0)}P^n,  \text{với } n = 0, 1, 2,...
\end{equation} 
$$

<!-- <a name="-Minh-hoa-1-qua-trinh-Markov"></a> -->
## 5. Illustration of a Markov Process
The initial problem begins is a market segmentation problem involving 2 stores.

Expansion: There are $N$ stores selling same type of product. Customers can make purchases at any of these $N$ stores based on their preferences, and they may switch from one to another, Stores use competitive strategies such as advertising, promotions,etc., to attract customers.

For example: Consider three stores and 1000 customers with the following initial distribution in January ($X_0$):
$$
\begin{equation} \notag
[P(X_0 = 1) = 20\% \quad P(X_0 = 2) = 50\% \quad P(X_0 = 3) = 30\%]
\end{equation} 
$$

After one month (1 cycle), data on customer purchases (which store they bought the product from) is collected. The table of results is as follows:

| Store  |January |From A | From B  | From C | To A | To B | To C | February |
|--|----------------|----------|----------|----------|-----------|-----------|-----------|---------------|
| A| 200            | 0        | 35       | 25      | 0         | 20        | 20        | 220            |
| B| 500            | 20       | 0       | 20       | 35         | 0        | 15        | 490           |
| C| 300            | 20       | 15       | 40       | 25        | 20        | 0        | 290           |

We have constructed the transition probability matrix as follows:
$$
\ {P} = \begin{pmatrix}
p_{11} = \frac{160}{200} = 0.800 & p_{12} = \frac{20}{200} = 0.100 & p_{13} = \frac{20}{200} = 0.100\\
p_{21} = \frac{35}{500} = 0.070 & p_{22} = \frac{450}{500} = 0.900 & p_{23} = \frac{15}{500} = 0.030 \\
p_{31} = \frac{25}{300} = 0.083 & p_{32} = \frac{20}{300} = 0.067 & p_{33} = \frac{255}{300} = 0.850 
\end{pmatrix}
$$

Market segmentation forecast for the future:
- What will the market look like in April? What will be the probability distribution for each store in April?
- How will the market look like one year from now?...

<!-- <a name="-Tong-ket"></a> -->
## 6. Conclusion
This post provided a brief introduction to Markov chain. In the next article, we will explore more details on this topic and have some Python coding exercises. 
Thanks for your reading!