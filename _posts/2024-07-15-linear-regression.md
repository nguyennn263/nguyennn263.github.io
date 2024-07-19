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
Suppose we have a dataset containing information about $m$ ouses that were sold in a specific area. The data includes the area in square meters ($a$) and the price ($b$). Our goal is to predict the price of a house given the information about its area.

Linear regression uses a model to solve this prediction problem. I will use a simple linear model, where:
$$y = f(x) =  \theta_0 + \theta_1 x$$

When we use the given dataset, we need to find $\theta_0$ and $\theta_1$ to determine the "best-fitting" model. Imagine the dataset contains many points; we want to find the line that best fits all of them.

However, we cannot always find a perfect linear relationship that fits all the data points in the dataset. Therefore, we aim to find $\theta_0$ and $\theta_1$ such that $y \approx f(x) $

## Least square error and Cost function
Let $\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \end{bmatrix} $.
Given the dataset, we have an example represented by a vector: $A_i = [1\quad a_1]$ and $b_i = [b_i]$
so that we can have a dot product: $A_i \theta = b_i$

Thus, we can build matrices and use matrix multiplication (dot product) to represent all of the equations above. Let:
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

It is common that  $A \theta = b$ has no solution. The reason is that there are to many equations and a line,  represented by $\theta$, cannot fit all of the points from dataset. Let $e = |A \theta - b|$ be the error, we must minimize $e$ as much as possible to find $\theta$, which is least square solution, and the predict $\hat{b} = A\theta$. The error will be $e = |\hat{b} - b|$

The format of the cost/loss fucntion is:
$$L(\theta) = ||e||^2  = ||A\theta-b||^2_2$$

The notation $||x||_2$ is the Euclidean norm. The loss function above represents the distance between vector $\hat{b} $ and $b$. Therefore, the loss fucntion is defined by the sum of square error:

$$L(\theta) = \sum^m_{i=1} (a_i\theta - b_i)$$

## Finding the Least square solution
Given matrix $A$ from database, we can obtain $C(A)$, the column space of A. Imagine that the column space of $A$ is a plane and vector $b$ is a point in $R_n$.

![Illustrative](1.png)
_Illustrative_

We have:
$$A\theta = 
\begin{bmatrix} 1 & a_1 \\ 1 & a_2 \\ \dots & \dots \\ 1 & a_m \end{bmatrix} \theta =
\begin{bmatrix} 1 \\ 1 \\ \dots \\ 1 \end{bmatrix} \theta_0 + \begin{bmatrix} a_1 \\ a_2 \\ \dots \\ a_m \end{bmatrix} \theta_1 = \hat{b}
$$.

Therefore $\hat{b}$ is a linear combination of $C(A)$. Finally, we must find the Least squares error which is $L(\theta)$ or distance between vector $\hat{b} $ and $b$. The distance of point $b$ and point $\hat{b} $ is minimum when $\hat{b}$ is projection of $b$ onto $C(A)$.

Because $\hat{b}$ is protjection of $b$ the error vector is perpendicular to the column space column space $C(A)$. Therefore, their dot product equals to $0$: $a_i^T(b_i - a_i\theta) = 0 $. This lead to $A^T (b- A\theta) = \overrightarrow{0}$.

We have:
$A^T (b- A\theta) = \overrightarrow{0} \Rightarrow A^TA \theta = A^Tb \Rightarrow \theta = (A^TA)^{-1}A^Tb$.

Therefore $\theta = (A^TA)^{-1}A^Tb$ is the least square solution.

## Discussion
If we have more details about the houses, such as the number of rooms, number of windows, etc., we can use the proof above to find the least squares solution. Thus, we can predict the price of a house given the number of rooms, windows, etc.

## Conclusion
This post provided a brief introduction to Linear Regression. In the next post, I will introduce Decision Trees.
Thank you for reading!