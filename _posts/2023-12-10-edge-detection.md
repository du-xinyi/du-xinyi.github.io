---
title: Edge Detection
description: 介绍图像边缘检测的评价标准，以及 Canny、Sobel、Laplacian 算子的原理和 OpenCV 用法
date: 2023-12-10 20:30:00 +0800
categories: [ Computer Vision, OpenCV ]
tags: [ OpenCV, Edge Detection, Image Processing ]
series: computer-vision
math: true
---

边缘是图像中灰度发生剧烈变化的位置，边缘检测通过计算图像的一阶或二阶导数找出这些位置。最优边缘检测的主要评价标准：

1. 低错误率：标识出尽可能多的实际边缘，同时尽可能的减少噪声产生的误报
2. 高定位性：标识出的边缘要与图像中的实际边缘尽可能接近
3. 最小响应：图像中的边缘只能标识一次，并且可能存在的图像噪声不应标识为边缘

导数运算会放大噪声，因此边缘检测之前通常先进行平滑滤波

## Sobel 算子

Sobel 算子用两个卷积核分别计算图像在 x 方向和 y 方向的一阶导数。假设被作用图像为$I$：

1. 分别在 x 和 y 两个方向求导

   $$
   G_x =
   \begin{bmatrix}
   -1 & 0 & +1 \\
   -2 & 0 & +2 \\
   -1 & 0 & +1 \\
   \end{bmatrix} * I
   ,\qquad
   G_y =
   \begin{bmatrix}
   -1 & -2 & -1 \\
   0 & 0 & 0 \\
   +1 & +2 & +1 \\
   \end{bmatrix} * I
   $$

   $G_x$反映水平方向的灰度变化，对竖直边缘响应强；$G_y$反映垂直方向的灰度变化，对水平边缘响应强

2. 在图像的每一点，结合以上两个结果求出梯度幅值和方向

   $$
   G = \sqrt{G_x^2 + G_y^2}
   ,\qquad
   \theta = \arctan\left(\frac{G_y}{G_x}\right)
   $$

   为了减少计算量，幅值也常用$G = \lvert G_x \rvert + \lvert G_y \rvert$近似

```c++
void Sobel( InputArray src, OutputArray dst, int ddepth,
            int dx, int dy, int ksize = 3,
            double scale = 1, double delta = 0,
            int borderType = BORDER_DEFAULT );
```
{: file='Sobel 算子'}

参数详解：

| 参数         | 类型          | 说明 |
| :----------- | :------------ | :--- |
| `src`        | `InputArray`  | 输入图像 |
| `dst`        | `OutputArray` | 目标图像，与源图片有一样的尺寸和通道数 |
| `ddepth`     | `int`         | 输出图像的深度，支持如下 src.depth() 和 ddepth 的组合：<br>若 src.depth() = CV_8U, 取 ddepth=-1,CV_16S,CV_32F,CV_64F<br>若 src.depth() = CV_16U 或 CV_16S, 取 ddepth =-1,CV_32F,CV_64F<br>若 src.depth() = CV_32F, 取 ddepth =-1,CV_32F,CV_64F<br>若 src.depth() = CV_64F, 取 ddepth = -1,CV_64F |
| `dx`         | `int`         | x 方向上的差分阶数 |
| `dy`         | `int`         | y 方向上的差分阶数 |
| `ksize`      | `int`         | 有默认值 3。Sobel 核的大小必须取 1，3，5 或 7 |
| `scale`      | `double`      | 计算导数值时可选的缩放因子，默认值是 1，表示默认情况下是没有应用缩放的 |
| `delta`      | `double`      | 表示在结果存入目标图（第二个参数 dst）之前可选的 delta 值，有默认值 0 |
| `borderType` | `int`         | 边界模式，默认值为 BORDER_DEFAULT |

导数有正有负，输入为`CV_8U`图像时如果`ddepth`取-1，负值会被截断为 0，丢失一半的边缘。应将`ddepth`设为`CV_16S`，再用`convertScaleAbs`取绝对值并转换回`CV_8U`

核大小为 3 时，Sobel 算子的近似误差较大，可以改用精度更高的`Scharr`算子，其参数与`Sobel`相同，但没有`ksize`参数

## Laplacian 算子

Laplacian 算子是二阶微分算子，定义为各方向二阶偏导数之和：

$$
Laplacian(f)=\frac{\partial^2f}{\partial x^2}+\frac{\partial^2f}{\partial y^2}
$$

在边缘处，一阶导数取得极值，二阶导数经过零点，因此可以通过二阶导数的过零点检测边缘。与 Sobel 算子不同，它不区分方向，一次运算即可得到各个方向的边缘，但对噪声更敏感。`ksize`为 1 时使用的卷积核为

$$
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0 \\
\end{bmatrix}
$$

```c++
void Laplacian( InputArray src, OutputArray dst, int ddepth,
                int ksize = 1, double scale = 1, double delta = 0,
                int borderType = BORDER_DEFAULT );
```
{: file='Laplacian 算子'}

参数详解：

| 参数         | 类型          | 说明 |
| :----------- | :------------ | :--- |
| `src`        | `InputArray`  | 输入图像 |
| `dst`        | `OutputArray` | 目标图像，与源图片有一样的尺寸和通道数 |
| `ddepth`     | `int`         | 输出图像的深度 |
| `ksize`      | `int`         | 用于计算二阶导数的滤波器的孔径尺寸，大小必须为正奇数，且有默认值 1 |
| `scale`      | `double`      | 计算拉普拉斯值的时候可选的比例因子，默认值是 1 |
| `delta`      | `double`      | 表示在结果存入目标图（第二个参数 dst）之前可选的 delta 值，有默认值 0 |
| `borderType` | `int`         | 边界模式，默认值为 BORDER_DEFAULT |

## Canny 算子

Canny 算子在梯度计算的基础上增加了后处理步骤，得到的是单像素宽、连续的二值边缘图。检测步骤：

1. 消除噪声  
   一般情况下，使用高斯平滑滤波器卷积降噪
2. 计算梯度幅值和方向  
   使用 Sobel 算子计算$G_x$和$G_y$，再求出每个像素的梯度幅值和方向，梯度方向近似到 0、45、90、135 度四个方向之一
3. 非极大值抑制  
   沿梯度方向比较相邻像素，只保留幅值为局部最大的像素。这一步排除非边缘像素，仅仅保留了一些细线条（候选边缘）
4. 滞后阈值  
   滞后阈值需要两个阈值（高阈值和低阈值）
   - 如果某一像素位置的幅值超过高阈值，该像素被保留为边缘像素
   - 如果某一像素位置的幅值小于低阈值，该像素被排除
   - 如果某一像素位置的幅值在两个阈值之间，该像素仅仅在连接到一个高于高阈值的像素时被保留

```c++
void Canny( InputArray image, OutputArray edges,
            double threshold1, double threshold2,
            int apertureSize = 3, bool L2gradient = false );
```
{: file='Canny 算子'}

参数详解：

| 参数           | 类型          | 说明 |
| :------------- | :------------ | :--- |
| `image`        | `InputArray`  | 输入图像，需为 8 位图像 |
| `edges`        | `OutputArray` | 输出的边缘图，为与源图片尺寸相同的 8 位单通道图像 |
| `threshold1`   | `double`      | 第一个滞后性阈值 |
| `threshold2`   | `double`      | 第二个滞后性阈值 |
| `apertureSize` | `int`         | 表示应用 Sobel 算子的孔径大小，其有默认值 3 |
| `L2gradient`   | `bool`        | 是否使用更精确的$\sqrt{G_x^2 + G_y^2}$计算梯度幅值，默认值 false 表示使用$\lvert G_x \rvert + \lvert G_y \rvert$ |

阈值 1 和阈值 2 两者的小者用于边缘连接，而大者用来控制强边缘的初始段。推荐的高低阈值比在 2:1 到 3:1 之间

## 使用示例

```c++
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat src = cv::imread("input.jpg");
    if (src.empty())
    {
        return -1;
    }

    // 转为灰度图并降噪
    cv::Mat gray, blurred;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);

    // Sobel：分别求x、y方向的梯度，取绝对值后合并
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, sobel;
    cv::Sobel(blurred, grad_x, CV_16S, 1, 0);
    cv::Sobel(blurred, grad_y, CV_16S, 0, 1);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);

    // Laplacian
    cv::Mat laplacian_16s, laplacian;
    cv::Laplacian(blurred, laplacian_16s, CV_16S, 3);
    cv::convertScaleAbs(laplacian_16s, laplacian);

    // Canny：低阈值50，高阈值150
    cv::Mat canny;
    cv::Canny(blurred, canny, 50, 150);

    cv::imwrite("sobel.jpg", sobel);
    cv::imwrite("laplacian.jpg", laplacian);
    cv::imwrite("canny.jpg", canny);

    return 0;
}
```
{: file='edge.cpp'}

```bash
g++ edge.cpp -o edge $(pkg-config --cflags --libs opencv4)
```
{: file='编译'}

三种算子在同一张测试图上的效果如下

![原图与 Sobel、Laplacian、Canny 三种算子的边缘检测结果对比](posts/2023-12-10-edge-detection/edge_compare.jpg)

| 算子      | 导数阶数 | 输出           | 特点                                           |
| :-------- | :------- | :------------- | :--------------------------------------------- |
| Sobel     | 一阶     | 梯度图（灰度） | 计算简单，可分别得到 x、y 方向的梯度，边缘较粗 |
| Laplacian | 二阶     | 梯度图（灰度） | 不区分方向，对噪声敏感                         |
| Canny     | 一阶     | 二值边缘图     | 边缘细且连续，抗噪声能力强，需要调节两个阈值   |
