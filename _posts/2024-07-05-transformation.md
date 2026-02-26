---
title: 齐次变换矩阵
date: 2024-07-05 12:30:00 +0800
categories: [ Notepad, Math ]
tags: [ Linear Algebra ]
math: True
---

# 齐次变换矩阵相关笔记

## 平移

将点$\left(x, y, z\right)$平移$t_x$，$t_y$，$t_z$单位

$$
T =
\begin{bmatrix}
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

## 旋转

旋转正负与右手定则一致：如果右手的拇指指向旋转轴的正方向，四指的弯曲方向表示正角度的旋转方向。即在右手坐标系下，逆时针旋转为正  

### 绕x轴旋转角度$\theta$

$$
R_x =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 绕y轴旋转角度$\theta$

$$
R_y =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 绕z轴旋转角度$\theta$

$$
R_z =
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 组合旋转

如果一个物体绕多个轴旋转，可以通过矩阵乘法将这些旋转矩阵组合起来。例如，如果一个物体先绕
z轴旋转角度$\alpha$，再绕y轴旋转角度$\beta$，最后绕x轴旋转角度$\gamma$，总的旋转矩阵R可以表示为：

$$
R = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

**矩阵乘法的顺序是从右到左，即最先旋转的矩阵在最右边**

即坐标系$\textbf{B}$的初始位姿与坐标系$\textbf{A}$重合，首先$\textbf{B}$相对于$\textbf{A}$的z轴顺时针旋转30度，再沿$\textbf{A}$的x轴正方向移动12个单位，并沿$\textbf{A}$的y轴正方向移动6个单位，点$\textbf{p}$在$\textbf{B}$的描述为$^{B}\textbf{p}=[3, 7, 0]^T$，则在$\textbf{A}$的描述$^{A}\textbf{p}$为

$$
^{A}_{B}R = R_{(z, 30^{\circ})} = 
\begin{bmatrix}
\cos(30^{\circ}) & -\sin(30^{\circ}) & 0 & 0 \\
\sin(30^{\circ}) & \cos(30^{\circ}) & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
^{A}P_{B_o} = 
\begin{bmatrix}
12 \\
6 \\
0
\end{bmatrix}
$$

$$
^{A}P = ^{A}_{B}R^{B}\textbf{p} + ^{A}P_{B_o} =
\begin{bmatrix}
-0.902 \\
7.562 \\
0
\end{bmatrix}
+ 
\begin{bmatrix}
12 \\
6 \\
0
\end{bmatrix} = 
\begin{bmatrix}
11.098 \\
13.562 \\
0
\end{bmatrix}
$$

## 翻转

### 绕xy平面的翻转（翻转z轴）

$$
F_z =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 绕xz平面的翻转（翻转y轴）

$$
F_y =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 绕yz平面的翻转（翻转x轴）

$$
F_x =
\begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

## 缩放

将点$\left(x, y, z\right)$的x，y，z坐标值分别缩放$s_x$，$s_y$，$s_z$倍

$$
S =
\begin{bmatrix}
s_x & 0 & 0 & 0 \\
0 & s_y & 0 & 0 \\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$
