---
title: Matplotlib
date: 2024-01-06 12:30:00 +0800
categories: [ Notepad, Python ]
tags: [ Python ]
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

## 标题标签参数说明
### 显示位置(loc)
`title()`、`xlabel()`、`ylabel()`默认均为`'center'`

`title()`可设置为 `'left'`、`'right'`和`'center'`  
`xlabel()`可设置为: `'left'`、`'right'`和 `'center'`  
`ylabel()`可设置为: `'bottom'`、`'top'`和`'center'`  

### 字体大小(size)
值为整数

## 绘图
### 线图

```python
# 绘制单条线
plot([x], y, [fmt], *, data=None, **kwargs)
# 绘制多条线
plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
```
参数说明：
- x, y 点或线的节点，`x`为x轴数据，`y`为y轴数据，数据可以是列表或数组
- fmt 可选参数，定义图的基本属性如
  - `color` 线条颜色，可以是颜色名称、十六进制颜色、RGB值
  - `marker` 点型
  - `linestyle` 线型
- **kwargs 设置线条样式
  - `label` 线条标签 
  - `linewidth` 线宽
  - `markersize` 点标记大小
  - `markeredgecolor` 点标记边缘颜色
  - `markeredgewidth` 点标记边缘宽度
  - `markerfacecolor` 点标记填充颜色
  - `alpha` 透明度
  - `zorder` 控制绘图顺序

fmt接收的是每个属性的单个字母缩写，例如：
```python
plot(x, y, 'bo-')  # 蓝色圆点实线
```
若属性用的是全名则不能用`fmt`参数来组合赋值，应该用关键字参数对单个属性赋值，如
```python
plot(x,y,color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
plot(x,y,color='#900302',marker='+',linestyle='-')
```

### 网格线

```python
plt.grid(b=None, which='major', axis='both', **kwargs)
```

参数说明
- b 默认为`None`，可以设置布尔值，`true`为显示网格线，`false`为不显示
- which 应用更改的网格线，默认为`'major'`
  - `'major'` 只应用于主刻度网格线
  - `'minor'` 只应用于次刻度网格线
  - `'both` 同时应用于主刻度和次刻度网格线
- axis 设置显示哪个方向的网格线，可以是取`'both'`（默认），`'x'`或`'y'`，分别表示两个方向
- **kwargs 设置网格样式
  - `color` 网格线的颜色
  - `linestyle` 网格线的线型
  - `linewidth` 网格线的宽度，单位为点

### 散点图

```python
plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, 
vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False,
data=None, **kwargs)
```

参数说明
- x，y 输入数据，长度相同的数组，绘制散点图的数据点
- s 点的大小，默认为20，也可以是个数组，数组每个参数为对应点的大小
- c 点的颜色，默认为蓝色`'b'`，也可以是RGB或RGBA二维行数组
- marker 点的样式，默认为小圆圈`'o'`
- cmap 颜色映射，默认为`None`，用于将数据值映射到特定的颜色范围，`c`为浮点数数组的时生效。如果没有指定`cmap`，默认为`viridis`。显示需要使用`plt.colorbar()` 
- norm 归一化，默认为`None`，用于将数据值归一化到[0, 1]的范围，`c`为浮点数数组的时生效
- vmin，vmax 亮度设置，在`norm`参数存在时会忽略
- alpha 透明度设置，0-1 之间，默认为`None`，即不透明
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

### 柱形图

```python
plt.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
```

参数说明
- x 浮点型数组，柱形图的`x`轴数据
- height 浮点型数组，柱形图的高度
- width 浮点型数组，柱形图的宽度
- bottom 浮点型数组，底座的`y`坐标，默认为0。
- align 柱形图与`x`坐标的对齐方式，默认为`'center'`以`x`位置为中心。可选`'edge'`将柱形图的左边缘与`x`位置对齐。要对齐右边缘，可以传递负的`width`
- tick_label 每个柱体的标签名称
- alpha 柱状填充颜色的透明度
- edgecolor 柱体的边框颜色
- linewidth 柱体边框线的宽度
- 
水平方向的柱形图可以使用`plt.barh()`

### 饼图

```python
plt.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, 
shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, 
wedgeprops=None, textprops=None, center=0, 0, frame=False, rotatelabels=False, 
*, normalize=None, data=None)
```

参数说明
- x 浮点型数组或列表，用于绘制饼图的数据
- explode 数组，各个扇形之间的间隔，默认为0
- labels 列表，各个扇形的标签，默认为None
- colors 数组，各个扇形的颜色，默认为None
- autopct 设置饼图内各个扇形百分比显示格式，`%d%%`整数百分比，`%0.1f`一位小数，`%0.1f%%`一位小数百分比，`%0.2f%%`两位小数百分比
- labeldistance 标签标记的绘制位置，相对于半径的比例，默认值为`1.1`，如`<1`则绘制在饼图内侧
- pctdistance：：类似`labeldistance`，指定`autopct`的位置刻度，默认为`0.6`
- shadow 布尔值，设置饼图的阴影，默认为`False`，不设置阴影
- radius 设置饼图的半径，默认为`1`
- startangle 指定饼图的起始角度，默认为从`x`轴正方向逆时针画起，如设定`=90`则从`y`轴正方向画起
- counterclock 布尔值，指定是否逆时针绘制扇形，默认为`True`逆时针绘制
- wedgeprops 字典类型，默认为`None`。指定扇形的属性
  - linewidth 扇形边界线的宽度，默认为`0`
  - edgecolor 扇形边界线的颜色，默认为`white`
  - antialiased	布尔值，默认为`True`，对扇形边界进行抗锯齿处理
  - linestyle 边界线的线型，默认为`'-'`
  - hatch 扇形的填充图案（如`'/'`,`'\\'`,`'-'`,`' '`等，支持组合）
- textprops 字典类型，指定文本标签的属性，默认为`None`
  - fontsize 文本字体大小，可以是整数或字符串（如`'small'`,`'medium'`,`'large'`等）默认值由`Matplotlib`配置决定
  - color 文本颜色，默认为`'black'`，支持颜色名称、十六进制值、RGB值或RGBA值
  - weight 字体粗细，默认为`'normal'`，常用值有`'normal'`,`'bold'`,`'light'`
  - style 字体样式，默认为`'normal'`，常用值有`'normal'`,`'italic'`,`'oblique'`
  - family 字体系列，如`'serif'`,`'sans-serif'`,`'monospace'`等，使用`Matplotlib`默认字体
  - verticalalignment 文本垂直对齐方式，默认为`'center'`，可选`'center'`,`'top'`,`'bottom'`
  - horizontalalignment	文本水平对齐方式，默认为`'center'`，可选`'center'`,`'left'`,`'right'`
  - rotation 文本旋转角度，以度为单位，正值为逆时针，负值为顺时针，默认值由布局决定
- center 浮点型列表，指定饼图的中心位置，默认`(0,0)`
- frame 布尔类型，指定是否绘制饼图的边框，默认`False`不绘制带有表的轴框架
- rotatelabels 布尔类型，指定是否旋转文本标签，默认为`False`不旋转
- normalize 布尔类型，指定是否对输入的数据进行归一化处理，为`True`会将所有值的和归一化为`1.0`，然后按比例绘制饼图
- data 指定数据。如果设置`data`参数，则可以直接使用数据框中的列作为`x`、`labels`等参数的值，无需再次传递

除此之外，`pie()`函数还可以返回三个参数
- wedges 包含扇形对象的列表
- texts 一个包含文本标签对象的列表
- autotexts 一个包含自动生成的文本标签对象的列表

### 直方图

```python
plt.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, 
bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, 
log=False, color=None, label=None, stacked=False, **kwargs)
```

参数说明
- x 要绘制直方图的数据，一维数组或列表
- bins 直方图的箱数。默认为10
- range 直方图的值域范围，二元数组或列表。默认为`None`，即使用数据中的最小值和最大值
- density 是否将直方图归一化。默认为`False`，即直方图的高度为每个箱子内的样本数，而不是频率或概率密度
- weights 每个数据点的权重，默认为`None`
- cumulative 是否绘制累积分布图。默认为`False`
- bottom 直方图的起始高度。默认为`None`
- histtype 直方图的类型，默认为`'bar'`，可选
  - `'bar'` 传统柱状图
  - `'barstacked'` 堆叠柱状图
  - `'step'` 无填充的折线图
  - `'stepfilled'` 有填充的折线图
- align 直方图箱子的对齐方式，默认为`'mid'`，可以是`'left'`、`'mid'`、`'right'`
- orientation 直方图方向，默认为`'vertical'`，可以是`'vertical'`、`'horizontal'`
- rwidth 每个箱子的宽度，默认为`None`
- log 是否在y轴上使用对数刻度，默认为`False`
- color 直方图的颜色
- label 直方图的标签
- stacked 是否堆叠不同的直方图，默认为`False`
- **kwargs：可选参数
  - edgecolor 柱体边框颜色，默认为`'black'`
  - linewidth 柱体边框线宽（单位：点），默认为`1.5`
  - linestyle 柱体边框的线型，默认为`'--'`
  - alpha 透明度（0.0 到 1.0），值越小越透明，默认为`0.7`
  - hatch 填充图案
  - color 柱体颜色，可以是单个值或颜色列表，默认为`blue`

## 设置参数
### 颜色字符(color)

- `'b'`或`'blue'` 蓝色
- `'m'`或`'magenta'` 洋红色
- `'g'`或`'green'` 绿色
- `'y'`或`'yellow'` 黄色
- `'r'`或`'red'` 红色
- `'k'`或`'black'` 黑色
- `'w'`或`'white'` 白色
- `'c'`或`'cyan'` 青绿色
- `#008000` RGB颜色字符串。多条曲线不指定颜色时，会自动选择不同颜色
- `(0.1, 0.2, 0.5)`或 `(0.1, 0.2, 0.5, 0.8)` RGB或RGBA值
- `'0.5'` 灰色

### 线型参数(linestyle)

- `‐`或`'solid'` 实线
- `‐‐`或`'dashed'` 虚线
- `‐.`或`'dashdot'` 点划线
- `:'`或`'dotted'` 点线
- `None` 不划线

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

### 填充图案(hatch)

- `'/'`	斜线
- `'\'` 反斜线
- `'-'`	横线
- `'|'`	竖线
- `'+'`	十字线
- `'x'`	交叉线
- `'o'`	圆圈
- `'.'`	点
- `'*'`	星形
- `'-/--'`	可重复符号形成更密集的图案

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