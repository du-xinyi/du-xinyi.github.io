---
title: Image Filtering
description: 梳理线性滤波、非线性滤波与形态学滤波的原理，以及对应的 OpenCV 函数和使用示例
date: 2023-12-09 22:30:00 +0800
categories: [ Computer Vision, OpenCV ]
tags: [ OpenCV, Filtering, Image Processing ]
series: computer-vision
math: true
---

图像滤波，即在尽量保留图像细节特征的条件下对目标图像的噪声进行抑制，是图像预处理中不可缺少的操作，其处理效果的好坏将直接影响到后续图像处理和分析的有效性和可靠性

## 线性滤波

邻域算子（局部算子）是利用给定像素周围的像素值的决定此像素的最终输出值的一种算子。而线性邻域滤波是一种常用的邻域算子，像素的输出值取决于输入像素的加权和，具体过程如下图

![邻域算子：输入图像与核卷积得到输出图像](posts/2023-12-09-image-filtering/邻域算子.png)

左边图像与中间图像的卷积产生右边图像。目标图像中蓝色标记的像素是利用原图像中红色标记的像素计算得到的  
线性滤波处理的输出像素值`g(i,j)`是输入像素值`f(i+k,j+l)`的加权和 :

$$
g(i,j) = \sum_{k,l}f(i+k,j+l)h(k,l)
$$

其中的`h(k,l)`为“核”，是滤波器的加权系数，即滤波器的滤波系数（严格来说上式是相关运算，OpenCV 的`filter2D`也是如此；卷积需先将核翻转，对称核两者结果相同）。可以简单写作

$$
g = f\otimes h
$$

其中`f`表示输入像素值，`h`表示加权系数“核“，`g`表示输出像素值

### 方框滤波（boxFilter）

方框滤波所用的核表示如下

$$
K = \alpha\begin{bmatrix}
1 & 1 & 1 & ... & 1 & 1 \\
1 & 1 & 1 & ... & 1 & 1 \\
..&...&...&.....&...&.. \\
1 & 1 & 1 & ... & 1 & 1 \\
\end{bmatrix}
$$

其中

$$
\alpha = \left\{\begin{matrix}
\frac{1}{ksize.width*ksize.height} & normalize = true \\
1 & normalize = false
\end{matrix}
\right.
$$

当 normalize=true 的时候，方框滤波就变成了均值滤波。也就是说，均值滤波是方框滤波归一化（normalized）后的特殊情况  
而非归一化（Unnormalized）的方框滤波用于计算每个像素邻域内的积分特性，比如密集光流算法（dense optical flow
algorithms）中用到的图像导数的协方差矩阵（covariance matrices of image derivatives）

```c++
void boxFilter(InputArray src,OutputArray dst, int ddepth, 
               Size ksize, Point anchor=Point(-1,-1), 
               bool normalize=true, 
               int borderType=BORDER_DEFAULT );
```
{: file='方框滤波'}

参数详解：

| 参数         | 类型          | 说明 |
| :----------- | :------------ | :--- |
| `src`        | `InputArray`  | 输入图像。该函数对通道是独立处理的，且可以处理任意通道数的图片，但需要注意，待处理的图片深度应该为`CV_8U`, `CV_16U`, `CV_16S`, `CV_32F`以及`CV_64F`之一 |
| `dst`        | `OutputArray` | 目标图像，与源图片有一样的尺寸和类型 |
| `ddepth`     | `int`         | 输出图像的深度，-1 代表使用原图深度，即`src.depth()` |
| `ksize`      | `Size`        | 内核的大小。一般用`Size(w, h)`来表示内核的大小（其中，`w`为像素宽度，`h`为像素高度） |
| `anchor`     | `Point`       | 表示锚点（即被平滑的那个点），默认为`Point(-1,-1)`。如果这个点坐标是负值的话，就表示取核的中心为锚点 |
| `normalize`  | `bool`        | 默认值为`true`，一个标识符，表示内核是否被其区域归一化 |
| `borderType` | `int`         | 用于推断图像外部像素的某种边界模式。有默认值`BORDER_DEFAULT` |

### 均值滤波（blur）

均值滤波本身存在着固有的缺陷，即它不能很好地保护图像细节，在图像去噪的同时也破坏了图像的细节部分，从而使图像变得模糊，不能很好地去除噪声点

```c++
void blur(InputArray src, OutputArray dst, 
          Size ksize, Point anchor=Point(-1,-1), 
          int borderType=BORDER_DEFAULT );
```
{: file='均值滤波'}

参数详解：

| 参数         | 类型          | 说明 |
| :----------- | :------------ | :--- |
| `src`        | `InputArray`  | 输入图像。该函数对通道是独立处理的，且可以处理任意通道数的图片，但需要注意，待处理的图片深度应该为`CV_8U`, `CV_16U`, `CV_16S`, `CV_32F`以及`CV_64F`之一 |
| `dst`        | `OutputArray` | 目标图像，与源图片有一样的尺寸和类型 |
| `ksize`      | `Size`        | 内核的大小。一般用`Size(w, h)`来表示内核的大小（其中，`w`为像素宽度，`h`为像素高度） |
| `anchor`     | `Point`       | 表示锚点（即被平滑的那个点），默认为`Point(-1,-1)`。如果这个点坐标是负值的话，就表示取核的中心为锚点 |
| `borderType` | `int`         | 用于推断图像外部像素的某种边界模式。有默认值`BORDER_DEFAULT` |

### 高斯滤波（GaussianBlur）

高斯滤波就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。高斯滤波的具体操作是：用一个模板（或称卷积核、掩模）扫描图像中的每一个像素，用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值  
高斯滤波器是一类根据高斯函数的形状来选择权值的线性平滑滤波器。高斯平滑滤波器对于抑制服从正态分布的噪声非常有效。一维零均值高斯函数为：

$$
G(x) = exp(\frac{-x^{2}}{2\sigma^{2}})
$$

其中，高斯分布参数 Sigma 决定了高斯函数的宽度。对于图像处理来说，常用二维零均值离散高斯函数作平滑滤波器  
二维高斯函数为：

$$
G_{0}(x,y) = A\exp\left(-\frac{(x-u_{x})^2}{2\sigma_{x}^2}-\frac{(y-u_{y})^2}{2\sigma_{y}^2}\right),\quad A=\frac{1}{2\pi\sigma_{x}\sigma_{y}}
$$

```c++
void GaussianBlur(InputArray src,OutputArray dst, Size ksize, 
                  double sigmaX, double sigmaY=0, 
                  int borderType=BORDER_DEFAULT );
```
{: file='高斯滤波'}

参数详解：

| 参数         | 类型          | 说明 |
| :----------- | :------------ | :--- |
| `src`        | `InputArray`  | 输入图像。该函数对通道是独立处理的，且可以处理任意通道数的图片，但需要注意，待处理的图片深度应该为`CV_8U`, `CV_16U`, `CV_16S`, `CV_32F`以及`CV_64F`之一 |
| `dst`        | `OutputArray` | 目标图像，与源图片有一样的尺寸和类型 |
| `ksize`      | `Size`        | 高斯内核的大小。其中 ksize.width 和 ksize.height 可以不同，但他们都必须为正数和奇数。或者，它们可以是零，由 sigma 计算而来 |
| `sigmaX`     | `double`      | 表示高斯核函数在 X 方向的的标准偏差 |
| `sigmaY`     | `double`      | 表示高斯核函数在 Y 方向的的标准偏差。若 sigmaY 为零，就将其设为 sigmaX，如果 sigmaX 和 sigmaY 都是 0，那么就由 ksize.width 和 ksize.height 计算出来 |
| `borderType` | `int`         | 用于推断图像外部像素的某种边界模式。有默认值`BORDER_DEFAULT` |

## 非线性滤波

### 中值滤波（medianBlur）

中值滤波（Median filter）是一种典型的非线性滤波技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值，该方法在去除脉冲噪声、椒盐噪声的同时又能保留图像边缘细节。
中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术，其基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，让周围的像素值接近的真实值，从而消除孤立的噪声点，对于斑点噪声（speckle
noise）和椒盐噪声（salt-and-pepper noise）来说尤其有用，因为它不依赖于邻域内那些与典型值差别很大的值。中值滤波器在处理连续图像窗函数时与线性滤波器的工作方式类似，但滤波过程却不再是加权运算。
中值滤波在一定的条件下可以克服常见线性滤波器如最小均方滤波、方框滤波器、均值滤波等带来的图像细节模糊，而且对滤除脉冲干扰及图像扫描噪声非常有效，也常用于保护边缘信息

```c++
void medianBlur( InputArray src, OutputArray dst, int ksize );
```
{: file='中值滤波'}

参数详解：

| 参数    | 类型          | 说明 |
| :------ | :------------ | :--- |
| `src`   | `InputArray`  | 输入图像。该函数对通道是独立处理的，填入 1、3 或者 4 通道的图像，当 ksize 为 3 或者 5 的时候，图像深度需为`CV_8U`，`CV_16U`，或`CV_32F`，而对于较大孔径尺寸的图片，只能是`CV_8U` |
| `dst`   | `OutputArray` | 目标图像，与源图片有一样的尺寸和类型 |
| `ksize` | `int`         | 孔径的线性尺寸。必须是大于 1 的奇数 |

#### 中值滤波与均值滤波器比较

- 优势：在均值滤波器中，由于噪声成分被放入平均计算中，所以输出受到了噪声的影响，但是在中值滤波器中，由于噪声成分很难选上，所以几乎不会影响到输出。因此同样用 3x3 区域进行处理，中值滤波消除的噪声能力更胜一筹
- 劣势：中值滤波需要对邻域内的像素排序，计算量比均值滤波大；窗口较大时会抹掉细线、尖角等细节

### 双边滤波（bilateralFilter）

双边滤波（Bilateral filter）是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折衷处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的。具有简单、非迭代、局部的特点。
双边滤波器的好处是可以做边缘保存（edge
preserving），一般过去用的维纳滤波或者高斯滤波去降噪，都会较明显地模糊边缘，对于高频细节的保护效果并不明显。双边滤波器顾名思义比高斯滤波多了一个高斯方差 sigma－r，它是基于像素值相似度的高斯函数，所以在边缘附近，与当前像素灰度差别较大的像素不会太多影响到边缘上的像素值，这样就保证了边缘附近像素值的保存。但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波

在双边滤波器中，输出像素的值依赖于邻域像素值的加权值组合：

$$
g(i,j)=\frac{\sum{_{k,l}}f(k,l)w(i,j,k,l)}{\sum{_{k,l}}w(i,j,k,l)}
$$

加权系数 w(i,j,k,l) 取决于定义域核和值域核的乘积  
其中定义域核表示如下

$$
d(i,j,k,l)=exp\Big(-\frac{(i-k)^2+(j-l)^2}{2\sigma^2_d} \Big)
$$

值域核表示为

$$
r(i,j,k,l)=exp\Big(-\frac{\parallel f(i,j)-f(k,l)\parallel^2}{2\sigma^2_r} \Big)
$$

两者相乘后，就会产生依赖于数据的双边滤波权重函数

$$
w(i,j,k,l)=exp\Big(-\frac{(i-k)^2+(j-l)^2}{2\sigma^2_d} - \frac{\parallel f(i,j)-f(k,l)\parallel^2}{2\sigma^2_r} \Big)
$$

```c++
void bilateralFilter( InputArray src, OutputArray dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType = BORDER_DEFAULT );
```
{: file='双边滤波'}

参数详解：

| 参数         | 类型          | 说明 |
| :----------- | :------------ | :--- |
| `src`        | `InputArray`  | 输入图像。需要为 8 位或者浮点型单通道、三通道的图像，三通道时按颜色向量的距离计算值域权重（各通道并非独立处理） |
| `dst`        | `OutputArray` | 目标图像，与源图片有一样的尺寸和类型 |
| `d`          | `int`         | 表示在过滤过程中每个像素邻域的直径。如果这个设为非正数，那么会从第五个参数`sigmaSpace`来计算 |
| `sigmaColor` | `double`      | 颜色空间滤波器的 sigma 值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域 |
| `sigmaSpace` | `double`      | 坐标空间中滤波器的 sigma 值，坐标空间的标注方差。其数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当 d>0，d 指定了邻域大小且与 sigmaSpace 无关。否则，d 正比于 sigmaSpace |
| `borderType` | `int`         | 用于推断图像外部像素的某种边界模式。有默认值`BORDER_DEFAULT` |

## 形态学滤波

### 膨胀（dilate）

膨胀就是求局部最大值的操作，核 B 与图形卷积，即计算核 B 覆盖的区域的像素点的最大值，并把这个最大值赋值给参考点指定的像素。这样就会使图像中的高亮区域逐渐增长

![膨胀操作示意](posts/2023-12-09-image-filtering/dilate.png)

```c++
void dilate( InputArray src, OutputArray dst, InputArray kernel,
             Point anchor = Point(-1,-1), int iterations = 1,
             int borderType = BORDER_CONSTANT,
             const Scalar& borderValue = morphologyDefaultBorderValue() );
```
{: file='膨胀'}

参数详解：

| 参数          | 类型            | 说明 |
| :------------ | :-------------- | :--- |
| `src`         | `InputArray`    | 输入图像，即源图像，填 Mat 类的对象即可。图像通道的数量可以是任意的，但图像深度应为 CV_8U，CV_16U，CV_16S，CV_32F 或 CV_64F 其中之一 |
| `dst`         | `OutputArray`   | 即目标图像，与源图片有一样的尺寸和类型 |
| `kernel`      | `InputArray`    | 膨胀操作的核。若为空矩阵`Mat()`，表示使用参考点位于中心的 3x3 核。一般使用函数`getStructuringElement`配合这个参数的使用。`getStructuringElement`函数会返回指定形状和尺寸的结构元素（内核矩阵）。其中，`getStructuringElement`函数的第一个参数表示内核的形状，可以选择如下三种形状<br>矩形：MORPH_RECT<br>交叉形：MORPH_CROSS<br>椭圆形：MORPH_ELLIPSE |
| `anchor`      | `Point`         | 锚的位置，其有默认值（-1，-1），表示锚位于中心 |
| `iterations`  | `int`           | 迭代使用 dilate() 函数的次数，默认值为 1 |
| `borderType`  | `int`           | 用于推断图像外部像素的某种边界模式。有默认值 BORDER_CONSTANT |
| `borderValue` | `const Scalar&` | 当边界为常数时的边界值，有默认值 morphologyDefaultBorderValue() |

### 腐蚀（erode）

膨胀和腐蚀是相反的一对操作，所以腐蚀就是求局部最小值的操作

![腐蚀操作示意](posts/2023-12-09-image-filtering/erode.png)

```c++
void erode( InputArray src, OutputArray dst, InputArray kernel,
            Point anchor = Point(-1,-1), int iterations = 1,
            int borderType = BORDER_CONSTANT,
            const Scalar& borderValue = morphologyDefaultBorderValue() );
```
{: file='腐蚀'}

参数详解：

| 参数          | 类型            | 说明 |
| :------------ | :-------------- | :--- |
| `src`         | `InputArray`    | 输入图像，即源图像，填 Mat 类的对象即可。图像通道的数量可以是任意的，但图像深度应为 CV_8U，CV_16U，CV_16S，CV_32F 或 CV_64F 其中之一 |
| `dst`         | `OutputArray`   | 即目标图像，与源图片有一样的尺寸和类型 |
| `kernel`      | `InputArray`    | 腐蚀操作的核。若为空矩阵`Mat()`，表示使用参考点位于中心的 3x3 核 |
| `anchor`      | `Point`         | 锚的位置，其有默认值（-1，-1），表示锚位于中心 |
| `iterations`  | `int`           | 迭代使用 erode() 函数的次数，默认值为 1 |
| `borderType`  | `int`           | 用于推断图像外部像素的某种边界模式。有默认值 BORDER_CONSTANT |
| `borderValue` | `const Scalar&` | 当边界为常数时的边界值，有默认值 morphologyDefaultBorderValue() |

### 开运算（Opening Operation）

先腐蚀后膨胀的过程称为开运算。
开运算可以用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积

### 闭运算（Closing Operation）

先膨胀后腐蚀的过程称为闭运算。
闭运算能够填充物体内部的小孔洞和细小的缝隙

### 形态学梯度（MorphologicalGradient）

形态学梯度为膨胀图与腐蚀图之差

### 顶帽（Top Hat）

顶帽运算为原图像与开运算的结果图之差。
开运算会去掉比核小的明亮区域，因此从原图中减去开运算后的图，得到的效果图突出了比周围更明亮的小区域，且这一操作和选择的核的大小相关。
顶帽运算往往用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取

### 黑帽（Black Hat）

黑帽运算为闭运算的结果图与原图像之差。
黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，且这一操作和选择的核的大小相关。
黑帽运算用来分离比邻近点暗一些的斑块

### 命中遗漏变换（Hit Miss）

命中遗漏变换主要用于检测图像中特定的形状或模式。该操作可以用于二值图像，其中目标是找到与指定的模式匹配的像素

![命中遗漏变换的运算过程](posts/2023-12-09-image-filtering/hitmiss.png)

使用 B1 对 A 进行腐蚀得到结果图像 C1；
使用 B2 对 A 的补集进行腐蚀得到结果图像 C2；
取 C1 和 C2 的交集作为使用 B1 和 B2 对 A 进行命中遗漏变换的结果

上述运算过程中 B1 为命中使用核，B2 为遗漏使用核

命中遗漏变换实际上是先在图像 A 中，寻找满足第一个结构元素 B1 模式的结构，找到之后相当于命中。然后用第二个结构元素 B2，直接在原图击中的位置进行匹配，如果不匹配，也就是遗漏。如果满足以上两点，就是要找的结构形状，把其与核锚点对应的中心像素置为 255，作为输出

### 核心 API

```c++
void morphologyEx( InputArray src, OutputArray dst,
                   int op, InputArray kernel,
                   Point anchor = Point(-1,-1), int iterations = 1,
                   int borderType = BORDER_CONSTANT,
                   const Scalar& borderValue = morphologyDefaultBorderValue() );
```
{: file='morphologyEx'}

参数详解：

| 参数          | 类型            | 说明 |
| :------------ | :-------------- | :--- |
| `src`         | `InputArray`    | 输入图像，即源图像，填 Mat 类的对象即可。图像通道的数量可以是任意的，但图像深度应为 CV_8U，CV_16U，CV_16S，CV_32F 或 CV_64F 其中之一 |
| `dst`         | `OutputArray`   | 即目标图像，与源图片有一样的尺寸和类型 |
| `op`          | `int`           | 表示形态学运算的类型，可以是如下之一的标识符<br>MORPH_ERODE 腐蚀<br>MORPH_DILATE 膨胀<br>MORPH_OPEN 开运算<br>MORPH_CLOSE 闭运算<br>MORPH_GRADIENT 形态学梯度<br>MORPH_TOPHAT 顶帽<br>MORPH_BLACKHAT 黑帽<br>MORPH_HITMISS 命中遗漏变换 |
| `kernel`      | `InputArray`    | 形态学运算的内核。若为空矩阵`Mat()`，表示使用参考点位于中心的 3x3 核 |
| `anchor`      | `Point`         | 锚的位置，其有默认值（-1，-1），表示锚位于中心 |
| `iterations`  | `int`           | 迭代使用函数的次数，默认值为 1 |
| `borderType`  | `int`           | 用于推断图像外部像素的某种边界模式。有默认值 BORDER_CONSTANT |
| `borderValue` | `const Scalar&` | 当边界为常数时的边界值，有默认值 morphologyDefaultBorderValue() |

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

    // 线性滤波
    cv::Mat box, mean, gaussian;
    cv::boxFilter(src, box, -1, cv::Size(5, 5));
    cv::blur(src, mean, cv::Size(5, 5));
    cv::GaussianBlur(src, gaussian, cv::Size(5, 5), 0);

    // 非线性滤波
    cv::Mat median, bilateral;
    cv::medianBlur(src, median, 5);
    cv::bilateralFilter(src, bilateral, 9, 75, 75);

    // 形态学滤波
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat dilated, eroded, opened, gradient;
    cv::dilate(src, dilated, kernel);
    cv::erode(src, eroded, kernel);
    cv::morphologyEx(src, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(src, gradient, cv::MORPH_GRADIENT, kernel);

    cv::imwrite("gaussian.jpg", gaussian);
    cv::imwrite("median.jpg", median);
    cv::imwrite("bilateral.jpg", bilateral);
    cv::imwrite("opened.jpg", opened);

    return 0;
}
```
{: file='filter.cpp'}

```bash
g++ filter.cpp -o filter $(pkg-config --cflags --libs opencv4)
```
{: file='编译'}

除`bilateralFilter`外，以上函数的`src`和`dst`可以是同一个对象，直接在原图上处理

对一张加入了高斯噪声和椒盐噪声的测试图分别进行滤波，效果如下

![原图、加噪声后的图像，以及均值、高斯、中值、双边滤波的结果对比](posts/2023-12-09-image-filtering/filter_compare.jpg)

- 均值滤波和高斯滤波把噪声抹平的同时也模糊了边缘，椒盐噪点被抹成了斑点
- 中值滤波基本去除了椒盐噪声，边缘仍然清晰，但细线变淡
- 双边滤波的边缘最清晰，平坦区域的高斯噪声被抹平，但椒盐噪点与周围差异过大，被当作边缘保留了下来

对一张带有白色噪点、黑色小孔和一条细线的二值图进行形态学运算，效果如下

![二值图的膨胀、腐蚀、开运算、闭运算与形态学梯度的结果对比](posts/2023-12-09-image-filtering/morphology_compare.png)

- 膨胀使白色区域扩大，小孔被填上，噪点也随之变大
- 腐蚀使白色区域缩小，噪点和细线消失，小孔变大
- 开运算去掉了白色噪点和细线，物体的大小基本不变
- 闭运算填上了较小的孔和缝隙，白色噪点仍然保留
- 形态学梯度得到物体的轮廓

## 滤波方法的选择

| 滤波方法 | 适用场景                                   | 特点                                 |
| :------- | :----------------------------------------- | :----------------------------------- |
| 均值滤波 | 对质量要求不高的快速平滑                   | 速度最快，边缘模糊明显               |
| 高斯滤波 | 高斯噪声，边缘检测等算法之前的预处理       | 平滑效果自然，边缘仍会模糊           |
| 中值滤波 | 椒盐噪声、脉冲噪声                         | 能去除孤立噪声点并保留边缘，速度较慢 |
| 双边滤波 | 需要在去噪的同时保留边缘，如人像磨皮       | 保边效果好，速度最慢                 |
| 开运算   | 去除二值图中的小噪点、断开细小的连接       | 基本不改变较大物体的面积             |
| 闭运算   | 填充二值图中物体内部的小孔洞、连接邻近区域 | 基本不改变较大物体的面积             |
