---
title: PyTorch Layers
description: 整理 PyTorch 卷积、转置卷积与池化层的输出尺寸计算方法及验证方式
date: 2024-07-10 12:30:00 +0800
categories: [ Development, Python ]
tags: [ PyTorch, Python, Deep Learning ]
math: true
---

## 输出大小计算

PyTorch 中二维图像张量的形状为$(N, C, H, W)$，依次为批大小、通道数、高度和宽度。下文用$D$表示通道数（深度），卷积和池化只改变$C$、$H$、$W$，不改变$N$

`kernel_size`、`stride`、`padding`、`dilation`可以是一个整数（高度和宽度方向相同），也可以是`(高度, 宽度)`形式的元组

### 卷积层

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
```

输入大小为$W \times H \times D$（宽度、高度和深度），卷积核大小为$K_w \times K_h$
，步幅（stride）为$S_w$和$S_h$，填充（padding）为$P_w$和$P_h$，膨胀（dilation）为$D_w$和$D_h$，输出深度（输出通道数）为$D_{out}$

- 输出宽度$W_{out}$：  
  $$
  W_{out} = \left\lfloor \frac{W + 2P_w - D_w \cdot (K_w - 1) - 1}{S_w} \right\rfloor + 1
  $$

- 输出高度$H_{out}$：  
  $$
  H_{out} = \left\lfloor \frac{H + 2P_h - D_h \cdot (K_h - 1) - 1}{S_h} \right\rfloor + 1
  $$

- 输出深度$D_{out}$等于卷积核的个数（输出通道数）

以$5 \times 5$的输入、`kernel_size=3`、`stride=2`、`padding=1`为例，卷积核每次移动 2 格，在每个方向上可以放置 3 个位置，输出为$3 \times 3$

![5x5 输入经过 3x3 卷积核、步幅 2、填充 1 得到 3x3 输出](posts/2024-07-10-pytorch-layers/conv_output.png)

常用的几种组合：

| kernel_size | stride | padding | 输出大小                 |
| :---------- | :----- | :------ | :----------------------- |
| 3           | 1      | 1       | 与输入相同               |
| 5           | 1      | 2       | 与输入相同               |
| 3           | 2      | 1       | 输入的一半（向上取整）   |
| 1           | 1      | 0       | 与输入相同，只改变通道数 |

`padding`也可以写成`'same'`，自动计算使输出与输入大小相同的填充（仅支持`stride=1`）

### 转置卷积层

```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1)
```

转置卷积用于上采样，输出大小的计算与卷积相反，额外的`output_padding`为$O_w$和$O_h$

- 输出宽度$W_{out}$：  
  $$
  W_{out} = (W - 1) \cdot S_w - 2P_w + D_w \cdot (K_w - 1) + O_w + 1
  $$

- 输出高度$H_{out}$：  
  $$
  H_{out} = (H - 1) \cdot S_h - 2P_h + D_h \cdot (K_h - 1) + O_h + 1
  $$

- 输出深度$D_{out}$等于输出通道数

卷积在步幅大于 1 时向下取整，多个不同的输入大小会得到相同的输出大小，`output_padding`用于在转置卷积时指定还原到其中的哪一个

### 池化层

#### 最大值池化

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)
```

`stride`默认为`None`，表示与`kernel_size`相同；`ceil_mode`设为`True`时，下面公式中的向下取整改为向上取整

输入大小为$W \times H \times D$（宽度、高度和深度），池化窗口大小为$K_w \times K_h$
，步幅（stride）为$S_w$和$S_h$，填充（padding）为$P_w$和$P_h$，膨胀（dilation）为$D_w$和$D_h$

- 输出宽度$W_{out}$：  
  $$
  W_{out} = \left\lfloor \frac{W + 2P_w - D_w \cdot (K_w - 1) - 1}{S_w} \right\rfloor + 1
  $$

- 输出高度$H_{out}$：  
  $$
  H_{out} = \left\lfloor \frac{H + 2P_h - D_h \cdot (K_h - 1) - 1}{S_h} \right\rfloor + 1
  $$

- 输出深度$D_{out}$等于输入深度$D$

#### 平均值池化

```python
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False)
```

输入大小为$W \times H \times D$（宽度、高度和深度），池化窗口大小为$K_w \times K_h$
，步幅（stride）为$S_w$和$S_h$，填充（padding）为$P_w$和$P_h$

- 输出宽度$W_{out}$：  
  $$
  W_{out} = \left\lfloor \frac{W + 2P_w - K_w}{S_w} \right\rfloor + 1
  $$

- 输出高度$H_{out}$：  
  $$
  H_{out} = \left\lfloor \frac{H + 2P_h - K_h}{S_h} \right\rfloor + 1
  $$

- 输出深度$D_{out}$等于输入深度$D$

#### 自适应平均值池化

```python
nn.AdaptiveAvgPool2d(output_size)
```

输入大小为$W \times H \times D$（宽度、高度和深度），目标输出大小为$W_{out} \times H_{out} \times D_{out}$

- 输出宽度$W_{out}$等于指定的目标宽度

- 输出高度$H_{out}$等于指定的目标高度

- 输出深度$D_{out}$等于输入深度$D$

`nn.AdaptiveAvgPool2d(1)`即全局平均池化，常用于分类网络中全连接层之前，使网络可以接受任意大小的输入

## 验证输出大小

实际搭建网络时，最直接的方法是构造一个与输入形状相同的张量，查看每一层输出的形状

```python
import torch
from torch import nn

x = torch.randn(1, 3, 224, 224)

conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
pool = nn.MaxPool2d(kernel_size=2)
up = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
gap = nn.AdaptiveAvgPool2d(1)

y = conv(x)
print(y.shape)        # torch.Size([1, 16, 112, 112])
print(pool(y).shape)  # torch.Size([1, 16, 56, 56])
print(up(y).shape)    # torch.Size([1, 8, 224, 224])
print(gap(y).shape)   # torch.Size([1, 16, 1, 1])
```
{: file='查看各层输出的形状'}
