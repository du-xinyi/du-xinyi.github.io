---
title: Matplotlib
date: 2024-10-05 12:30:00 +0800
categories: [ Notepad ]
tags: [ python ]
---

# matplotlib相关笔记

`matplotlib`是Python中绘图工具，使用时需要导入其中的`pyplot`模块

```python
import matplotlib.pyplot as plt
```
{: file='pyplot'}

常用的`pyplot`函数：
- figure()：创建画布
- title()：设置标题
- xlabel()：横坐标名称
- ylabel()：纵坐标名称
- legend()：添加图例
- plot()：绘制线图和散点图
- scatter()：绘制散点图
- bar()：绘制垂直条形图和水平条形图
- hist()：绘制直方图
- pie()：绘制饼图
- grid(): 绘制网格线
- imshow()：显示图像
- subplots()：创建子图

## 绘图参数说明
```python
# 绘制单条线
plot([x], y, [fmt], *, data=None, **kwargs)
# 绘制多条线
plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
```
参数说明：
- x, y：点或线的节点，`x`为x轴数据，`y`为y轴数据，数据可以是列表或数组
- fmt：可选参数，定义图的基本属性如：颜色（color），点型（marker），线型（linestyle）

### 颜色字符(color)
- `b` 蓝色
- `m` 洋红色
- `g` 绿色
- `y` 黄色
- `r` 红色
- `k` 黑色
- `w` 白色
- `c` 青绿色
- `#008000` RGB颜色字符串。多条曲线不指定颜色时，会自动选择不同颜色

### 线型参数(linestyle)
- `‐` 实线
- `‐‐` 虚线
- `‐.` 点划线
- `:'` 点线
- `None` 不划线

### 线宽度(linewidth)
值为浮点数，如`1`、`2.0`、`5.67`等

### 标记字符(marker)
- `'.'`	点
- `','`	像素点
- `'o'`	实心圆
- `'v'`	下三角
- `'^'`	上三角
- `'<'`	左三角
- `'>'`	右三角
- `'1'`	下三叉
- `'2'`	上三叉
- `'3'`	左三叉
- `'4'`	右三叉
- `'8'`	八角形
- `'s'`	正方形
- `'p'`	五边形
- `'P'`	加号（填充）
- `'*'`	星号
- `'h'`	六边形 1
- `'H'`	六边形 2
- `'+'`	加号
- `'x'`	乘号 x
- `'X'`	乘号 x (填充)
- `'D'`	菱形
- `'d'`	瘦菱形
- `'|'`	竖线
- `'_'`	横线
- `0` 左横线
- `1` 右横线
- `2` 上竖线
- `3` 下竖线
- `4` 左箭头
- `5` 右箭头
- `6` 上箭头
- `7` 下箭头
- `8` 左箭头 (中间点为基准)
- `9` 右箭头 (中间点为基准)
- `10` 上箭头 (中间点为基准)
- `11` 下箭头 (中间点为基准)
- `None`,`' '`,`''`	没有任何标记
- `$...$` 渲染指定的字符。例如`'$f$'`以字母`f`为标记

fmt接收的是每个属性的单个字母缩写，例如：
```python
plot(x, y, 'bo-')  # 蓝色圆点实线
```
若属性用的是全名则不能用`fmt`参数来组合赋值，应该用关键字参数对单个属性赋值，如
```python
plot(x,y,color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
plot(x,y,color='#900302',marker='+',linestyle='-')
```

## 标题标签参数说明
### 显示位置(loc)
`title()`、`xlabel()`、`ylabel()`默认均为`'center'`

`title()`可设置为 `'left'`、`'right'`和`'center'`  
`xlabel()`可设置为: `'left'`、`'right'`和 `'center'`  
`ylabel()`可设置为: `'bottom'`、`'top'`和`'center'`  

### 字体大小(size)
值为整数

## 设置网格线

```python
plt.grid(b=None, which='major', axis='both', **kwargs)
```

参数说明
- b 默认为`None`，可以设置布尔值，`true`为显示网格线，`false`为不显示
- which 可选值有 'major'、'minor' 和 'both'，默认为'major'，表示应用更改的网格线
- axis 设置显示哪个方向的网格线，可以是取`'both'`（默认），`'x'`或`'y'`，分别表示两个方向
- **kwargs 设置网格样式，可以是网格线的颜色(color)，样式(linestyle)和宽度(linewidth)

## 绘制多图
`subplot()`在绘图时需要指定位置，`subplots()`可以一次生成多个，在调用时只需要调用生成对象的`ax`即可

### subplot()

```python
plt.subplot(nrows, ncols, index, **kwargs)
plt.subplot(pos, **kwargs)
plt.subplot(**kwargs)
plt.subplot(ax)
```

以上函数将整个绘图区域分成`nrows`行和`ncols`列，然后从左到右，从上到下的顺序对每个子区域进行编号`1...N`，左上的子区域的编号为1、右下的区域编号为N，编号可以通过参数`index`来设置

### subplots()

```python
plt.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, 
subplot_kw=None, gridspec_kw=None, **fig_kw)
```

参数说明
- nrows 默认为1，设置图表的行数
- ncols 默认为1，设置图表的列数
- sharex、sharey 设置`x`、`y`轴是否共享属性，默认为`false`，可设置为`'none'`、`'all'`、`'row'`或`'col'`。`none`每个子图的`x`轴或`y`轴都是独立的，`'all'`所有子图共享`x`轴或`y`轴，`'row'`设置每个子图行共享一个`x`轴或`y`轴，`'col'`每个子图列共享一个`x`轴或`y`轴
- squeeze 默认为`True`，表示额外的维度从返回的Axes(轴)对象中挤出，对于`N*1`或`1*N`个子图，返回一个1维数组，对于`N*M`，`N>1`和`M>1`返回一个2维数组。如果设置为`False`，则不进行挤压操作，返回一个元素为Axes实例的2维数组，即使它最终是1x1。
- subplot_kw 把字典的关键字传递给`add_subplot()`来创建每个子图
- gridspec_kw 把字典的关键字传递给`GridSpec`构造函数创建子图放在网格里
- **fig_kw：把详细的关键字参数传给`figure()` 

## 绘制散点图

```python
plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, 
vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False,
data=None, **kwargs)
```

参数说明
- x，y 输入数据，长度相同的数组，绘制散点图的数据点
- s 点的大小，默认20，也可以是个数组，数组每个参数为对应点的大小
- c 点的颜色，默认蓝色`'b'`，也可以是RGB或RGBA二维行数组
- marker 点的样式，默认小圆圈`'o'`
- cmap 颜色映射，默认`None`，用于将数据值映射到特定的颜色范围，`c`为浮点数数组的时生效。如果没有指定`cmap`，默认为`viridis`。显示需要使用`plt.colorbar()` 
- norm 归一化，默认`None`，用于将数据值归一化到[0, 1]的范围，`c`为浮点数数组的时生效
- vmin，vmax 亮度设置，在`norm`参数存在时会忽略
- alpha 透明度设置，0-1 之间，默认`None`，即不透明
- linewidths 标记点的长度
- edgecolors 颜色或颜色序列，默认为`'face'`，可选值有`'face'`,`'none'`,`None`
- plotnonfinite 布尔值，设置是否使用非限定的`c`(`inf`,`-inf`或`nan`)绘制点
- **kwargs 其他参数

颜色条参数可以为

| 颜色名称 |   反转色   |                 描述                 |
| :------: | :--------: | :----------------------------------: |
|  Accent  |  Accent_r  |        强调配色，适合分类数据        |
|  Blues   |  Blues_r   |        蓝色渐变，适合连续数据        |
|   BrBG   |   BrBG_r   |  棕色到蓝绿色渐变，用于两极数据对比  |
|   BuGn   |   BuGn_r   |     蓝色到绿色渐变，用于连续数据     |
|   BuPu   |   BuPu_r   |     蓝色到紫色渐变，用于连续数据     |
|  CMRmap  |  CMRmap_r  |         基于 CMR 的颜色映射          |
|  Dark2   |  Dark2_r   |       暗色系配色，用于分类数据       |
|   GnBu   |   GnBu_r   |     绿色到蓝色渐变，用于连续数据     |
|  Greens  |  Greens_r  |        绿色渐变，适合连续数据        |
|  Greys   |  Greys_r   |  灰色渐变，适合背景或无显著差异数据  |
|   OrRd   |   OrRd_r   |     橙色到红色渐变，适合强度数据     |
| Oranges  | Oranges_r  |        橙色渐变，适合连续数据        |
|   PRGn   |   PRGn_r   |       紫色到绿色渐变，两极对比       |
|  Paired  |  Paired_r  |         配对色，适合分类数据         |
| Pastel1  | Pastel1_r  |        柔和配色，适合分类数据        |
| Pastel2  | Pastel2_r  |        柔和配色，适合分类数据        |
|   PiYG   |   PiYG_r   |      粉色到黄绿色渐变，两极对比      |
|   PuBu   |   PuBu_r   |     紫色到蓝色渐变，用于连续数据     |
|  PuBuGn  |  PuBuGn_r  |    紫色到蓝绿色渐变，用于连续数据    |
|   PuOr   |   PuOr_r   |       紫色到橙色渐变，两极对比       |
|   PuRd   |   PuRd_r   |     紫色到红色渐变，用于连续数据     |
| Purples  | Purples_r  |        紫色渐变，适合连续数据        |
|   RdBu   |   RdBu_r   |       红色到蓝色渐变，两极对比       |
|   RdGy   |   RdGy_r   |     红色到灰色渐变，用于两极对比     |
|   RdPu   |   RdPu_r   |     红色到紫色渐变，用于连续数据     |
|  RdYlBu  |  RdYlBu_r  |   红色到黄色再到蓝色，用于两极对比   |
|  RdYlGn  |  RdYlGn_r  |   红色到黄色再到绿色，用于两极对比   |
|   Reds   |   Reds_r   |     红色渐变，适合强度或热量数据     |
|   Set1   |   Set1_r   |       配色集合1，适合分类数据        |
|   Set2   |   Set2_r   |       配色集合2，适合分类数据        |
|   Set3   |   Set3_r   |       配色集合3，适合分类数据        |
| Spectral | Spectral_r |     多色渐变，用于连续或离散数据     |
|  Wistia  |  Wistia_r  |       黄色调渐变，适合特殊效果       |
|   YlGn   |   YlGn_r   |     黄色到绿色渐变，用于连续数据     |
|  YlGnBu  |  YlGnBu_r  |    黄色到蓝绿色渐变，用于连续数据    |
|  YlOrBr  |  YlOrBr_r  |     黄色到橙棕渐变，用于连续数据     |
|  YlOrRd  |  YlOrRd_r  |  黄色到橙红渐变，用于强度或热量数据  |
| viridis  | viridis_r  | 蓝绿黄色调，常用于连续数据，视觉友好 |
|  plasma  |  plasma_r  |     高对比蓝黄渐变，适合连续数据     |
| inferno  | inferno_r  |     高对比橙黄渐变，适合连续数据     |
|  magma   |  magma_r   |     高对比紫黄渐变，适合连续数据     |
| cividis  | cividis_r  |         色盲友好的蓝绿色渐变         |
|  winter  |  winter_r  |       蓝绿色渐变，适合连续数据       |

## 柱形图

```python
plt.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
```

参数说明
- x 浮点型数组，柱形图的`x`轴数据
- height 浮点型数组，柱形图的高度
- width 浮点型数组，柱形图的宽度
- bottom 浮点型数组，底座的`y`坐标，默认0。
- align 柱形图与`x`坐标的对齐方式，默认为`'center'`以`x`位置为中心。可选`'edge'`将柱形图的左边缘与`x`位置对齐。要对齐右边缘，可以传递负的`width`
- tick_label 每个柱体的标签名称
- alpha 柱状填充颜色的透明度
- edgecolor 柱体的边框颜色
- linewidth 柱体边框线的宽度
- 
水平方向的柱形图可以使用`plt.barh()`