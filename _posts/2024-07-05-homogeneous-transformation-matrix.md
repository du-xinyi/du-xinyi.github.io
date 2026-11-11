---
title: Homogeneous Transformation Matrix
description: 整理三维空间中的平移、旋转、镜像、缩放矩阵，齐次变换矩阵的组成、求逆与链式变换，以及欧拉角和四元数
date: 2024-07-05 12:30:00 +0800
categories: [ Mathematics, Linear Algebra ]
tags: [ Linear Algebra, Coordinate Transform, Robotics ]
math: true
---

三维空间中的点$\left(x, y, z\right)$写成齐次坐标$\left[x, y, z, 1\right]^T$后，平移、旋转、缩放都可以用一个$4 \times 4$矩阵的乘法表示，多个变换可以通过矩阵相乘合并为一个矩阵

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

### 绕 x 轴旋转角度$\theta$

$$
R_x =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 绕 y 轴旋转角度$\theta$

$$
R_y =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 绕 z 轴旋转角度$\theta$

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
z 轴旋转角度$\alpha$，再绕 y 轴旋转角度$\beta$，最后绕 x 轴旋转角度$\gamma$，总的旋转矩阵 R 可以表示为：

$$
R = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\gamma & -\sin\gamma & 0 \\
0 & \sin\gamma & \cos\gamma & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
\cos\beta & 0 & \sin\beta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\beta & 0 & \cos\beta & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 & 0 \\
\sin\alpha & \cos\alpha & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

**矩阵乘法的顺序是从右到左，即最先旋转的矩阵在最右边**（此处各次旋转均绕固定坐标轴；若绕物体自身的动轴旋转，乘法顺序相反）

## 齐次变换矩阵

坐标系$\textbf{B}$相对于坐标系$\textbf{A}$的位姿由两部分组成：旋转矩阵$^{A}\_{B}R$描述$\textbf{B}$的姿态，位置向量$^{A}\textbf{p}\_{B\_o}$描述$\textbf{B}$的原点在$\textbf{A}$中的位置。点$\textbf{p}$在两个坐标系中的坐标满足

$$
^{A}\textbf{p} = {}^{A}_{B}R \, ^{B}\textbf{p} + {}^{A}\textbf{p}_{B_o}
$$

把旋转和平移合并到一个$4 \times 4$矩阵中，就得到齐次变换矩阵

$$
^{A}_{B}T =
\begin{bmatrix}
^{A}_{B}R & ^{A}\textbf{p}_{B_o} \\
\textbf{0}^T & 1 \\
\end{bmatrix}
,\qquad
\begin{bmatrix}
^{A}\textbf{p} \\
1 \\
\end{bmatrix}
= {}^{A}_{B}T
\begin{bmatrix}
^{B}\textbf{p} \\
1 \\
\end{bmatrix}
$$

### 示例

坐标系$\textbf{B}$的初始位姿与坐标系$\textbf{A}$重合，首先$\textbf{B}$相对于$\textbf{A}$的 z 轴逆时针旋转 30 度，再沿$\textbf{A}$的 x 轴正方向移动 12 个单位，并沿$\textbf{A}$的 y 轴正方向移动 6 个单位，点$\textbf{p}$在$\textbf{B}$的描述为$^{B}\textbf{p}=[3, 7, 0]^T$，求其在$\textbf{A}$的描述$^{A}\textbf{p}$

$$
^{A}_{B}R = R_{(z, 30^{\circ})} =
\begin{bmatrix}
\cos(30^{\circ}) & -\sin(30^{\circ}) & 0 \\
\sin(30^{\circ}) & \cos(30^{\circ}) & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
,\qquad
^{A}\textbf{p}_{B_o} =
\begin{bmatrix}
12 \\
6 \\
0
\end{bmatrix}
$$

$$
\begin{bmatrix}
^{A}\textbf{p} \\
1 \\
\end{bmatrix}
=
\begin{bmatrix}
0.866 & -0.5 & 0 & 12 \\
0.5 & 0.866 & 0 & 6 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
3 \\
7 \\
0 \\
1 \\
\end{bmatrix}
=
\begin{bmatrix}
11.098 \\
13.562 \\
0 \\
1 \\
\end{bmatrix}
$$

![坐标系 A、坐标系 B 与点 p 的位置关系](posts/2024-07-05-homogeneous-transformation-matrix/frames.png)

### 逆变换

旋转矩阵是正交矩阵，其逆等于转置，因此齐次变换矩阵的逆不需要按一般矩阵求逆，可以直接写出

$$
^{B}_{A}T = \left(^{A}_{B}T\right)^{-1} =
\begin{bmatrix}
^{A}_{B}R^T & -^{A}_{B}R^T \, ^{A}\textbf{p}_{B_o} \\
\textbf{0}^T & 1 \\
\end{bmatrix}
$$

### 链式变换

已知$\textbf{B}$相对于$\textbf{A}$的变换$^{A}\_{B}T$和$\textbf{C}$相对于$\textbf{B}$的变换$^{B}\_{C}T$，则$\textbf{C}$相对于$\textbf{A}$的变换为两者相乘，相邻的上下标可以"消去"

$$
^{A}_{C}T = {}^{A}_{B}T \, ^{B}_{C}T
$$

## 姿态的其他表示

旋转矩阵用 9 个数表示 3 个自由度，存在冗余，实际使用中常与以下两种表示互相转换

### 欧拉角

把姿态分解为绕三个坐标轴的三次旋转。以常用的 ZYX 顺序（偏航 yaw、俯仰 pitch、滚转 roll）为例，依次绕自身的 z 轴、y 轴、x 轴旋转$\psi$、$\theta$、$\phi$，对应的旋转矩阵为

$$
R = R_z(\psi) R_y(\theta) R_x(\phi)
$$

欧拉角直观、只需 3 个数，但存在两个问题：

- 旋转顺序和绕固定轴还是自身轴有多种约定，不同的库和资料并不一致，使用前必须确认
- 中间的旋转角为$\pm 90^{\circ}$时，第一次和第三次旋转的轴重合，丢失一个自由度，称为万向锁

### 四元数

绕单位向量$\textbf{n} = [n\_x, n\_y, n\_z]^T$旋转角度$\theta$，可以用一个单位四元数表示

$$
q = \begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix}
= \begin{bmatrix} \cos\frac{\theta}{2} \\ n_x\sin\frac{\theta}{2} \\ n_y\sin\frac{\theta}{2} \\ n_z\sin\frac{\theta}{2} \end{bmatrix}
,\qquad w^2 + x^2 + y^2 + z^2 = 1
$$

对应的旋转矩阵为

$$
R =
\begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2) \\
\end{bmatrix}
$$

四元数只需 4 个数，没有万向锁，便于插值，是 ROS、Eigen 等存储姿态的常用形式。$q$与$-q$表示同一个旋转；不同的库中分量的存储顺序可能是$(w, x, y, z)$或$(x, y, z, w)$

## 镜像

### 关于 xy 平面的镜像（z 坐标取反）

$$
F_z =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 关于 xz 平面的镜像（y 坐标取反）

$$
F_y =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

### 关于 yz 平面的镜像（x 坐标取反）

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

将点$\left(x, y, z\right)$的 x，y，z 坐标值分别缩放$s_x$，$s_y$，$s_z$倍

$$
S =
\begin{bmatrix}
s_x & 0 & 0 & 0 \\
0 & s_y & 0 & 0 \\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

## 代码实现

用 NumPy 构造齐次变换矩阵，并验证前面的示例

```python
import numpy as np


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def make_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


T_ab = make_transform(rot_z(np.deg2rad(30)), [12, 6, 0])

p_b = np.array([3, 7, 0, 1])  # 齐次坐标
p_a = T_ab @ p_b
print(np.round(p_a, 3))  # [11.098 13.562  0.     1.   ]

T_ba = invert_transform(T_ab)
print(np.round(T_ba @ p_a, 3))  # [3. 7. 0. 1.]
print(np.allclose(T_ba, np.linalg.inv(T_ab)))  # True
```
{: file='齐次变换矩阵'}

旋转矩阵、欧拉角、四元数之间的转换可以使用`scipy`

```python
import numpy as np
from scipy.spatial.transform import Rotation

rotation = Rotation.from_euler("ZYX", [30, 0, 0], degrees=True)  # 绕自身轴的ZYX顺序

print(np.round(rotation.as_matrix(), 3))
print(np.round(rotation.as_quat(), 3))  # [0.    0.    0.259 0.966]，顺序为(x, y, z, w)
print(rotation.as_euler("ZYX", degrees=True))
```
{: file='姿态表示之间的转换'}
