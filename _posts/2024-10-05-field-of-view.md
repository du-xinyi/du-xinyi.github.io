---
title: Field of View
description: 介绍标准镜头与鱼眼镜头在不同投影模型下，由焦距和传感器尺寸计算视场角的方法
date: 2024-10-05 12:30:00 +0800
categories: [ Computer Vision, Camera ]
tags: [ Camera, Optics ]
math: true
---

镜头的视场角（Field of View, FOV）可以通过镜头的焦距`f`和传感器的宽度`W`高度`H`对角线长度`D`来计算

## 标准镜头

水平方向视场角：

$$
\theta_h = 2 \times \arctan(\frac{W}{2f})
$$

垂直方向视场角：

$$
\theta_v = 2 \times \arctan(\frac{H}{2f})
$$

对角线视场角：

$$
\theta_d = 2 \times \arctan(\frac{D}{2f})
$$

例如焦距$f = 4\,mm$、传感器宽度$W = 6.4\,mm$的镜头，水平视场角为

$$
\theta_h = 2 \times \arctan(\frac{6.4}{2 \times 4}) \approx 77.3^{\circ}
$$

## 鱼眼镜头

标准镜头遵循针孔模型，入射角$\alpha$的光线成像在距图像中心$r = f\tan\alpha$的位置，入射角接近$90^{\circ}$时$r$趋于无穷大，因此无法覆盖很大的视场角。鱼眼镜头改用其他投影模型，把大角度的光线压缩到有限的传感器范围内，视场角的计算取决于镜头采用的投影模型

| 投影模型     | 成像公式                       | 视场角                                    |
| :----------- | :----------------------------- | :---------------------------------------- |
| 等距投影     | $r = f\alpha$                  | $\theta = \frac{W}{f}$（弧度）            |
| 等立体角投影 | $r = 2f\sin(\frac{\alpha}{2})$ | $\theta = 4 \times \arcsin(\frac{W}{4f})$ |
| 体视投影     | $r = 2f\tan(\frac{\alpha}{2})$ | $\theta = 4 \times \arctan(\frac{W}{4f})$ |
| 正交投影     | $r = f\sin\alpha$              | $\theta = 2 \times \arcsin(\frac{W}{2f})$ |

表中的$W$换成传感器的高度$H$或对角线长度$D$，即可得到垂直或对角线视场角。等距投影在计算机视觉中使用最多（如 OpenCV 的`fisheye`模型），等立体角投影常见于摄影用的鱼眼镜头。实际镜头并不严格符合某一种模型，精确的视场角需要通过相机标定得到

![左：标准镜头视场角的几何关系；右：不同投影模型下像高随入射角的变化](posts/2024-10-05-field-of-view/fov.png)

左图为标准镜头的成像几何。右图中针孔模型的像高在入射角接近$90^{\circ}$时迅速增大，而四种鱼眼投影模型的像高始终有限，因此可以覆盖$180^{\circ}$甚至更大的视场角

以等立体角投影为例

水平方向视场角：

$$
\theta_h = 4 \times \arcsin(\frac{W}{4f})
$$

垂直方向视场角：

$$
\theta_v = 4 \times \arcsin(\frac{H}{4f})
$$

对角线视场角：

$$
\theta_d = 4 \times \arcsin(\frac{D}{4f})
$$
