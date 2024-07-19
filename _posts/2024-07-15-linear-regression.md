---
title: Linear Regression
date: 2024-07-15
categories: [Knowledge]
tags: [machine_learning, linear_algebra, least_square, supervised_learning]
description:  First machine learning algorithm
math: true
published: true
media_subpath: /assets/linear-regression/
# image:
    # path: alt-img.png
---

In this post, I will introduce Linear regression - the first Machine learning algorithm I've learn.

## Introduction
Suppose we have a dataset containing information of m houses which were sold in a specific area. The data contains area in meter square ($a$) and the price ($b$). Our goal is predict the price of house providing that we have infomation about area. 

Linear regression uses a model to solve prediction problem. I will use the simple linear model, let: 
$$y = f(x) =  \theta_0 + \theta_1 x$$

When we use the given dataset, we must find the $\theta_0$ and $\theta_1$ to find the "fittest" model. Imagining that given dataset contains a lot of points, we can find the line across all of them.

But the problem is we cannot always find a linear relationship that fit all the data in dataset, we will find the $\theta_0$ and $\theta_1$ so that $y \approx f(x) $

## Least square error and Cost function
Let $\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \end{bmatrix} $.
Given dataset, we have an example which is a vector: $A_i = [1\quad a_1]$ and $b_i = [b_i]$
so that we can have a dot product: $A_i \theta = b_i$

Therefore, we can build matrices and use matrix-multiply (dot product) to present all of equations above. Let
$A = 
\begin{bmatrix} 
1 & a_1 \\
1 & a_2 \\
\dots & \dots \\
1 & a_n
\end{bmatrix} $, 
$\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \end{bmatrix} $,
$b = 
\begin{bmatrix} 
b_1 \\
b_2 \\
\dots \\
b_n
\end{bmatrix} $ so that we have $A \theta = b$.

It is often that  $A \theta = b$ has no solution. The reason is there are to many equations and a line, which is presented by $\theta$, cannot fit all of points from dataset. Let $e = |A \theta - b|$ is error, we must reduce $e$ as small as possible so we have the $\theta$, which is least square solution, and the predict $\hat{b} = A\theta$. The error will be $e = |\hat{b} - b|$

The format of the cost/loss fucntion is:
$$L(\theta) = ||e||^2  = ||A\theta-b||^2_2$$

The notation $||x||_2$ is Euclidean norm. The loss function above presents the distance of two vector $\hat{b} $ and $b$. Therefore, the loss fucntion is defined by the sum of square error:

$$L(\theta) = \sum^m_{i=1} (a_i\theta - b_i)$$

## Finding the Least square solution
Given matrix $A$ from database, we can have the $C(A)$, column space of A. Imagining that the column space of $A$ is a plane and vector $b$ is a point in $R_n$.

![Illustrative](1.png)
_Illustrative_

We have:
$$A\theta = 
\begin{bmatrix} 1 & a_1 \\ 1 & a_2 \\ \dots & \dots \\ 1 & a_m \end{bmatrix} \theta =
\begin{bmatrix} 1 \\ 1 \\ \dots \\ 1 \end{bmatrix} \theta_0 + \begin{bmatrix} a_1 \\ a_2 \\ \dots \\ a_m \end{bmatrix} \theta_1 = \hat{b}
$$.

Therefore $\hat{b}$ is a linear combination of $C(A)$. Finally, we must find the Least square error which is $L(x)$ or distance of two vector $\hat{b} $ and $b$. The distance of point $b$ and point $\hat{b} $ is minimum when $\hat{b}$ is projection of $b$ in $C(A)$.

Because $\hat{b}$ is protjection of $b$ so the error vector is perpendicular to column space $C(A)$. Therefore, their dot products equal to 0: $a_i^T(b_i - a_i\theta) = 0 $. This lead to $A^T (b- A\theta) = \overrightarrow{0}$.

We have:
$A^T (b- A\theta) = \overrightarrow{0} \Rightarrow A^TA \theta = A^Tb \Rightarrow \theta = (A^TA)^{-1}A^Tb$.

Therefore $\theta = (A^TA)^{-1}A^Tb$ is the Least square solution.

## Discussion
If we have more detail about houses such as: number of rooms, number of windows,... we can using the proof above to find the Least square solution. Therefore, we can predict the price of house providing the number of rooms, windows,...

## Conclusion
This post provided a brief introduction to Linear Regression. In the next post, I will introduce you about Decision tree.
Thanks for your reading!