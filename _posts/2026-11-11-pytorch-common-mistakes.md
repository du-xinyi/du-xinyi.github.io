---
title: PyTorch Common Mistakes
description: 整理 PyTorch 中不报错但结果错误的常见问题，包括损失函数的输入、梯度清零、广播、eval 模式、计算图被切断以及标签的类型与范围
date: 2026-11-11 14:10:00 +0800
categories: [ Deep Learning, PyTorch ]
tags: [ PyTorch, Python, Deep Learning, Debugging ]
series: deep-learning
math: true
---

下面几种错误都不会抛出异常，程序可以正常跑完，只是损失不下降或者结果比预期差。训练出现问题时整体的排查顺序见[PyTorch Debugging](/posts/pytorch-debugging/)

文中的输出是在 CPU 上得到的。浮点运算的顺序与线程数、硬件有关，换一台机器运行时，训练过程中的具体数值可能略有不同，损失震荡、发散的例子差别会更大，但现象是一样的

## softmax 与 CrossEntropyLoss

`nn.CrossEntropyLoss`的输入是没有经过归一化的 logits，它在内部已经包含了`log_softmax`。在模型的最后再加一个`softmax`，相当于做了两次。`softmax`的输出在 0 到 1 之间，再做一次`softmax`后各类别的概率相差不大，即使预测完全正确，10 个类别时损失也不会低于$-\ln\frac{e}{e + 9} \approx 1.46$，同时梯度变得很小，训练变慢

```python
import torch
from torch import nn

criterion = nn.CrossEntropyLoss()

# 一个完全正确且非常确定的预测
logits = torch.full((1, 10), -20.0)
logits[0, 3] = 20.0
target = torch.tensor([3])
print(criterion(logits, target).item())                 # 0.0
print(criterion(logits.softmax(dim=1), target).item())  # 1.4611501693725586，再准确也降不下去
```
{: file='多做一次 softmax 的影响'}

[PyTorch Debugging](/posts/pytorch-debugging/)中单批次过拟合的第三种情况就是这个错误，损失停在 1.49。`nn.BCEWithLogitsLoss`同理，它的输入是没有经过`sigmoid`的 logits；需要概率时（推理、可视化）在模型之外再调用`softmax`或`sigmoid`

## 忘记清零梯度

`backward()`计算出的梯度会累加到参数的`.grad`上，而不是覆盖。每一步更新之前不调用`optimizer.zero_grad()`，用来更新参数的就是之前所有步的梯度之和

```python
import torch
from sklearn.datasets import load_digits
from torch import nn

layer = nn.Linear(3, 1)
for i in range(3):
    layer(torch.ones(1, 3)).sum().backward()
    print(layer.weight.grad)  # 每次backward都累加到上一次的梯度上

digits = load_digits()
x = torch.tensor(digits.data, dtype=torch.float32) / 16
y = torch.tensor(digits.target)
criterion = nn.CrossEntropyLoss()

for zero_grad in (True, False):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for step in range(300):
        loss = criterion(model(x), y)
        if zero_grad:
            optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"zero_grad={zero_grad}  " + "  ".join(f"{losses[i]:.3f}" for i in (0, 19, 49, 99, 199, 299)))
```
{: file='忘记清零梯度的影响'}

最后两行依次为第 1、20、50、100、200、300 步的损失

```text
tensor([[1., 1., 1.]])
tensor([[2., 2., 2.]])
tensor([[3., 3., 3.]])
zero_grad=True  2.314  0.397  0.088  0.033  0.010  0.004
zero_grad=False  2.314  0.610  2.783  1.609  2.566  3.980
```
{: file='输出结果'}

测试中发现，把上面的优化器换成不带动量的`SGD`并使用较小的学习率时，不清零梯度的损失仍然能够下降，表现得像是加了动量，因此这个错误不一定能从损失曲线上直接看出来，要靠检查代码确认

## 损失函数中的广播

回归任务中，模型的输出形状为$(N, 1)$，而标签的形状通常是$(N,)$。两者相减时会广播成$(N, N)$，每个预测值与所有标签都算了一次误差。此时让损失最小的解是所有样本都输出标签的均值，模型什么也学不到

```python
import torch
from torch import nn

torch.manual_seed(0)
x = torch.rand(100, 1)
y = 2 * x.squeeze(1) + 1 + 0.05 * torch.randn(100)  # 形状为(100,)，真实关系为y = 2x + 1

for fix in (False, True):
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for step in range(500):
        pred = model(x)  # 形状为(100, 1)
        if fix:
            pred = pred.squeeze(1)  # 形状为(100,)
        loss = nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"fix={fix}  loss {loss.item():.4f}  w {model.weight.item():.3f}  b {model.bias.item():.3f}")

print((model(x) - y).shape)  # torch.Size([100, 100])，(100, 1)与(100,)广播成了100x100
print(f"y的均值 {y.mean().item():.3f}  y的方差 {y.var(unbiased=False).item():.4f}")
```
{: file='MSE 损失中的广播'}

```text
fix=False  loss 0.3338  w 0.000  b 1.977
fix=True  loss 0.0020  w 1.966  b 1.017
torch.Size([100, 100])
y的均值 1.977  y的方差 0.3338
```
{: file='输出结果'}

错误的写法中权重为 0、偏置等于标签的均值，损失等于标签的方差。`nn.MSELoss`遇到这种情况会给出一条警告，但只显示一次，容易被其他输出淹没；自己写的`((pred - y) ** 2).mean()`则没有任何提示

```text
UserWarning: Using a target size (torch.Size([100])) that is different to the input size (torch.Size([100, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
```
{: file='警告信息'}

修复方法是让两者形状一致：`pred.squeeze(1)`或`y.unsqueeze(1)`。不要使用不带参数的`squeeze()`，批大小为 1 时它会把批的维度也去掉

## 忘记切换 eval 模式

`BatchNorm`和`Dropout`在训练和推理时的行为不同，由`model.train()`和`model.eval()`切换。两者的原理见[Normalization and Regularization](/posts/normalization-and-regularization/)

```python
import torch
from sklearn.datasets import load_digits
from torch import nn

torch.manual_seed(0)
digits = load_digits()
x = torch.tensor(digits.data, dtype=torch.float32) / 16
y = torch.tensor(digits.target)
x_train, y_train, x_val, y_val = x[:1397], y[:1397], x[1397:], y[1397:]

model = nn.Sequential(
    nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
for step in range(200):
    loss = criterion(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@torch.no_grad()
def accuracy(batch_size):
    correct = 0
    for i in range(0, len(x_val), batch_size):
        out = model(x_val[i:i + batch_size])
        correct += (out.argmax(dim=1) == y_val[i:i + batch_size]).sum().item()
    return correct / len(x_val)

model.eval()
print(torch.equal(model(x_val[:4]), model(x_val[:4])))  # True
print(f"eval   batch_size=256: {accuracy(256):.4f}  batch_size=4: {accuracy(4):.4f}")

model.train()
print(torch.equal(model(x_val[:4]), model(x_val[:4])))  # False，Dropout每次随机丢弃不同的神经元
print(f"train  batch_size=256: {accuracy(256):.4f}  batch_size=4: {accuracy(4):.4f}")

model.eval()
print(f"eval   batch_size=256: {accuracy(256):.4f}")  # 与第一次不同，running_mean已经被改动
```
{: file='验证时忘记调用 model.eval()'}

```text
True
eval   batch_size=256: 0.9300  batch_size=4: 0.9300
False
train  batch_size=256: 0.9250  batch_size=4: 0.8125
eval   batch_size=256: 0.9225
```
{: file='输出结果'}

- 推理模式下结果与批大小无关；训练模式下同一个模型的准确率随批大小变化，批大小为 4 时从 0.93 降到 0.81
- 训练模式下做验证还会改动`running_mean`和`running_var`，最后一行切回推理模式后准确率与第一次不同，验证集的统计量已经混进了模型
- 训练模式下`BatchNorm1d`遇到只有 1 个样本的批次会报错`Expected more than 1 value per channel when training`，这反而是容易发现的情况
- `torch.no_grad()`只是不记录计算图，不会改变这些层的行为，不能代替`model.eval()`；验证结束后回到训练时要重新调用`model.train()`

## 计算图被切断

`.item()`、`.detach()`、`.numpy()`、`torch.tensor(...)`得到的结果都不再属于计算图。在模型的中间使用了它们，这之前的所有参数都收不到梯度，但只要后面还有可以求导的参数，`backward()`就不会报错

```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
encoder = nn.Linear(8, 16)
head = nn.Linear(16, 2)
x = torch.randn(4, 8)
y = torch.tensor([0, 1, 0, 1])
criterion = nn.CrossEntropyLoss()

def check_grad(**modules):
    for name, module in modules.items():
        for param_name, param in module.named_parameters():
            norm = "None" if param.grad is None else f"{param.grad.norm().item():.4f}"
            print(f"{name}.{param_name:6s} grad: {norm}")

# 错误：中间用numpy做归一化，encoder到损失之间的计算图被切断
features = encoder(x).detach().numpy()
features = features / np.linalg.norm(features, axis=1, keepdims=True)
loss = criterion(head(torch.from_numpy(features)), y)
loss.backward()  # 不报错
check_grad(encoder=encoder, head=head)

# 正确：全程使用torch的运算
head.zero_grad()
features = encoder(x)
features = features / features.norm(dim=1, keepdim=True)
loss = criterion(head(features), y)
loss.backward()
check_grad(encoder=encoder, head=head)
```
{: file='检查每个参数是否收到梯度'}

```text
encoder.weight grad: None
encoder.bias   grad: None
head.weight grad: 0.3807
head.bias   grad: 0.0567
encoder.weight grad: 0.2696
encoder.bias   grad: 0.0372
head.weight grad: 0.3807
head.bias   grad: 0.0567
```
{: file='输出结果'}

- 调用一次`backward()`之后，梯度为`None`的参数就是没有连接到损失的参数。这个检查同样能发现定义了但没有在`forward`中使用的层
- 整个损失都脱离计算图时会报错`element 0 of tensors does not require grad and does not have a grad_fn`，比较容易发现
- 损失中的某一项使用了`.item()`时不会报错，这一项变成常数，不再起作用，例如`loss = ce + 0.1 * penalty.item()`
- 用`argmax`、取整、比较等不可导的运算得到的结果去计算损失，也收不到梯度

## 标签的类型与范围

`nn.CrossEntropyLoss`的标签是形状为$(N,)$、类型为`int64`的类别编号，取值范围为 0 到$K - 1$

```python
import torch
from torch import nn

torch.manual_seed(0)
logits = torch.randn(4, 3)  # 3个类别
criterion = nn.CrossEntropyLoss()

def run(name, target):
    try:
        print(f"{name}: {criterion(logits, target).item():.4f}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {e}")

run("正确        ", torch.tensor([0, 1, 2, 0]))
run("从1开始编号 ", torch.tensor([1, 2, 3, 1]))
run("浮点类型    ", torch.tensor([0.0, 1.0, 2.0, 0.0]))
run("形状为(N, 1)", torch.tensor([[0], [1], [2], [0]]))
run("独热编码    ", torch.eye(3)[[0, 1, 2, 0]])

# 二分类：BCEWithLogitsLoss的标签必须是0和1
logit = torch.tensor([2.0, -1.0, 3.0, -2.0])
bce = nn.BCEWithLogitsLoss()
print(bce(logit, torch.tensor([1.0, 0.0, 1.0, 0.0])).item())    # 标签为0/1
print(bce(logit, torch.tensor([1.0, -1.0, 1.0, -1.0])).item())  # 标签为-1/1，不报错，损失变成负数
print(bce(logit * 100, torch.tensor([1.0, -1.0, 1.0, -1.0])).item())  # 并且可以无限地降下去
```
{: file='标签的类型与范围'}

```text
正确        : 1.3928
从1开始编号 : IndexError: Target 3 is out of bounds.
浮点类型    : RuntimeError: expected target dtype to be Long or Byte, but got Float
形状为(N, 1): RuntimeError: 0D or 1D target tensor expected, multi-target not supported
独热编码    : 1.3928
0.1539263129234314
-0.5960736870765686
-75.0
```
{: file='输出结果'}

- 类型和形状不对会直接报错。形状为$(N, K)$的浮点标签会被当作各类别的概率，独热编码可以得到相同的结果
- 编号超出范围在 CPU 上的报错很清楚。在 GPU 上则是`CUDA error: device-side assert triggered`，并且报错的位置可能不是出错的那一行，把数据和模型放到 CPU 上运行一次就能看到真正的原因
- 标签从 1 开始编号、而输出层恰好有$K + 1$个输出时不会报错，只是第 0 类永远没有样本
- `nn.BCEWithLogitsLoss`不检查标签的取值，标签为 -1 和 1 时损失可以无限地降到负数，损失出现负值时先检查标签

## 汇总

| 问题                                 | 现象                                       | 修复                                                     |
| :----------------------------------- | :----------------------------------------- | :------------------------------------------------------- |
| `softmax`之后接`CrossEntropyLoss`    | 损失下降很慢，停在一个较高的值             | 模型直接输出 logits                                      |
| 忘记`optimizer.zero_grad()`          | 损失震荡、发散，或表现得像学习率过大       | 每一步`backward()`之前清零梯度                           |
| 预测值与标签的形状为$(N, 1)$和$(N,)$ | 损失停在标签的方差附近，输出接近标签的均值 | 计算损失前让两者形状一致                                 |
| 验证时没有调用`model.eval()`         | 验证结果偏低，并且随批大小变化             | 验证前`model.eval()`，训练前`model.train()`              |
| 计算图被切断                         | 部分参数不更新，损失下降慢或不下降         | 中间的运算全部使用`torch`的函数，检查`.grad`是否为`None` |
| 标签的取值不符合损失函数的要求       | 损失为负数，或某些类别永远预测不出来       | 检查标签的最小值、最大值和类型                           |
| 创建优化器之后重新创建了模型         | 损失完全不变                               | 先创建模型，再创建优化器                                 |
