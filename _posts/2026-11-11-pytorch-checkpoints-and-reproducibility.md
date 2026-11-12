---
title: PyTorch Checkpoints and Reproducibility
description: 整理 PyTorch 训练中检查点的保存与恢复、weights_only 的含义，以及固定随机种子和 GPU 上的不确定性
date: 2026-11-11 13:10:00 +0800
categories: [ Deep Learning, PyTorch ]
tags: [ PyTorch, Python, Deep Learning, Training ]
series: deep-learning
---

训练动辄几个小时到几天，中途可能被打断，训练完的结果也需要能够重新跑出来。这里整理检查点的保存与恢复，以及怎样让训练的结果可以复现，训练循环本身见[PyTorch Training](/posts/pytorch-training/)

## 保存与恢复

### 检查点

只保存模型参数，只够用来推理。要在中断之后接着训练，检查点中还需要优化器、调度器的状态和当前的轮数

| 内容                     | 不恢复的后果                                              |
| :----------------------- | :-------------------------------------------------------- |
| `model.state_dict()`     | 模型从头开始                                              |
| `optimizer.state_dict()` | 动量和 Adam 的统计量清零，恢复后的前几步不稳定            |
| `scheduler.state_dict()` | 学习率从头开始，回到初始的较大值或重新预热                |
| 轮数                     | 不知道还剩多少轮                                          |
| 随机数状态               | 之后的打乱顺序和`Dropout`与不中断时不同，结果不能逐位一致 |
| 最好的验证指标           | 恢复后第一轮就会把之前更好的模型覆盖掉                    |

下面的例子验证：训练 3 轮后保存，重新创建所有对象并从检查点恢复，再训练 3 轮，得到的结果与连续训练 6 轮完全相同

```python
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

digits = load_digits()
dataset = TensorDataset(torch.from_numpy(digits.data).float() / 16, torch.from_numpy(digits.target).long())
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
total_epochs = 6

def build():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs * len(loader))
    return model, optimizer, scheduler

def train(model, optimizer, scheduler, start_epoch, end_epoch):
    losses = []
    model.train()
    for epoch in range(start_epoch, end_epoch):
        total = 0.0
        for inputs, targets in loader:
            loss = F.cross_entropy(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        losses.append(round(total / len(loader), 6))
    return losses

def save_checkpoint(path, model, optimizer, scheduler, epoch):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),  # 动量、Adam的一阶和二阶矩估计
        "scheduler": scheduler.state_dict(),  # 当前走到了第几步
        "epoch": epoch,                       # 已经完成的轮数
        "rng": torch.get_rng_state(),         # 随机数生成器的状态，决定之后的打乱顺序和Dropout
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, restore_rng=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        torch.set_rng_state(checkpoint["rng"])
    return checkpoint["epoch"]

# 1. 不中断，连续训练6轮
model, optimizer, scheduler = build()
full = train(model, optimizer, scheduler, 0, total_epochs)
full_weight = model[0].weight.clone()
print("连续训练  ", full)

# 2. 训练3轮后保存，然后重新创建所有对象，从检查点恢复再训练3轮
model, optimizer, scheduler = build()
first = train(model, optimizer, scheduler, 0, 3)
save_checkpoint("checkpoint.pt", model, optimizer, scheduler, epoch=3)

model, optimizer, scheduler = build()
start_epoch = load_checkpoint("checkpoint.pt", model, optimizer, scheduler)
resumed = first + train(model, optimizer, scheduler, start_epoch, total_epochs)
print("中断后恢复", resumed)
print(resumed == full, torch.equal(model[0].weight, full_weight))  # True True

# 3. 只恢复模型，不恢复优化器和调度器
model, optimizer, scheduler = build()
model.load_state_dict(torch.load("checkpoint.pt", weights_only=True)["model"])
print("只恢复模型", first + train(model, optimizer, scheduler, 3, total_epochs))

# 4. 恢复了模型、优化器和调度器，但没有恢复随机数状态
model, optimizer, scheduler = build()
load_checkpoint("checkpoint.pt", model, optimizer, scheduler, restore_rng=False)
print("不恢复随机", first + train(model, optimizer, scheduler, 3, total_epochs))
```
{: file='保存检查点并恢复训练'}

```text
连续训练   [2.228253, 2.016717, 1.794023, 1.62148, 1.542186, 1.509996]
中断后恢复 [2.228253, 2.016717, 1.794023, 1.62148, 1.542186, 1.509996]
True True
只恢复模型 [2.228253, 2.016717, 1.794023, 1.560023, 1.306732, 1.122574]
不恢复随机 [2.228253, 2.016717, 1.794023, 1.622284, 1.534141, 1.509169]
```
{: file='输出结果'}

- 全部恢复后，每一轮的损失和最终的参数都与连续训练逐位相同
- 只恢复模型时，后 3 轮的损失明显不同：调度器从头开始，学习率回到了初始值。这里损失反而更低，只是因为这个例子中较大的学习率恰好更合适，训练过程已经不是原来计划的那一个
- 不恢复随机数状态时结果接近但不相同，一般可以接受；需要严格复现时才保存它
- 加载的顺序是先创建模型、优化器、调度器，再分别调用`load_state_dict`。优化器的状态会自动放到对应参数所在的设备上
- `map_location`指定把张量加载到哪个设备上。不指定时，在 GPU 上保存的张量会加载回 GPU，在没有 GPU 的机器上加载时必须指定`map_location="cpu"`

### weights_only

`torch.load`基于`pickle`，而`pickle`在加载时可以执行任意代码，来源不明的检查点文件是有风险的。`weights_only=True`只允许加载张量和基本的 Python 类型，是目前的默认值

```python
import numpy as np
import torch
from torch import nn

model = nn.Linear(4, 2)

# 只含张量、数字、字符串、列表、字典的检查点可以正常加载
torch.save({"model": model.state_dict(), "epoch": 3, "mean": [0.5]}, "plain.pt")
print(torch.load("plain.pt").keys())  # dict_keys(['model', 'epoch', 'mean'])

# 混入了numpy数组，或者保存的是整个模型对象，默认都会被拒绝
torch.save({"model": model.state_dict(), "mean": np.array([0.5])}, "mixed.pt")
torch.save(model, "whole.pt")
for path in ["mixed.pt", "whole.pt"]:
    try:
        torch.load(path)
    except Exception as error:
        print(path, type(error).__name__)  # UnpicklingError

# 确认文件来源可信时才关闭检查
print(type(torch.load("whole.pt", weights_only=False)).__name__)  # Linear
```
{: file='weights_only 的行为'}

- 检查点中只放`state_dict`、数字、字符串、列表和字典，numpy 数组先转成张量或列表
- 不要用`torch.save(model)`保存整个模型对象。它除了不能用`weights_only=True`加载，还依赖保存时的类定义和文件路径，代码改动后就可能加载失败
- 只有在确认文件来源可信时，才使用`weights_only=False`

## 可复现性

训练中的随机性来自参数初始化、数据打乱、数据增强、`Dropout`，以及 GPU 上部分算法本身的不确定性。固定随机种子可以消除前四项

```python
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 同时设置CPU和所有GPU的种子

class NoisyDataset(Dataset):
    """模拟带随机数据增强的数据集：返回下标和一个随机数"""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index, np.random.randint(0, 1000)

def run(seed):
    seed_everything(seed)
    model = nn.Linear(4, 2)  # 参数初始化用到随机数
    loader = DataLoader(NoisyDataset(), batch_size=4, shuffle=True, num_workers=2,
                        generator=torch.Generator().manual_seed(seed))  # 决定打乱顺序和各工作进程的种子
    batches = [(index.tolist(), noise.tolist()) for index, noise in loader]
    return round(model.weight.sum().item(), 4), batches

if __name__ == "__main__":
    print(run(0))
    print(run(0) == run(0))  # True，种子相同时，初始化、打乱顺序、工作进程中的随机数都相同
    print(run(0) == run(1))  # False
```
{: file='固定随机种子'}

```text
(-0.187, [([5, 3, 1, 7], [522, 297, 434, 198]), ([2, 6, 4, 0], [234, 407, 586, 805])])
True
False
```
{: file='输出结果'}

- 给`DataLoader`传入固定种子的`generator`后，打乱顺序和各个工作进程的种子都由它决定。每个工作进程中`torch`、`random`、`numpy`的种子由`DataLoader`自动设置为各不相同的值，不需要再用`worker_init_fn`手动设置
- 不传`generator`时，`DataLoader`从全局的随机数生成器取种子，只要之前调用过`torch.manual_seed`，结果同样可以复现
- 种子相同只保证同一台机器、同一个版本上的结果相同。不同的 PyTorch 版本、不同的硬件、CPU 与 GPU 之间，结果都可能有细微的差别

### GPU 上的不确定性

GPU 上即使种子相同，结果也可能不同：cuDNN 的部分卷积算法本身不保证每次的计算顺序一致，浮点数的舍入误差随之不同，经过多步训练后被放大。下面的脚本把同一个训练过程重复 3 次，输出来自一台 CPU 为 Intel Core Ultra 9 285K、GPU 为 GeForce RTX 5090 D（32 GB 显存）、内存为 64 GB 的机器

```python
import sys

import torch
import torch.nn.functional as F
from torch import nn

assert torch.cuda.is_available(), "这个例子需要GPU"

if len(sys.argv) > 1 and sys.argv[1] == "deterministic":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def run():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(32, 3, 64, 64, device="cuda")
    labels = torch.randint(0, 10, (32,), device="cuda")
    for _ in range(50):
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

print([round(run(), 6) for _ in range(3)])  # 种子相同的3次训练，最后一步的损失
```
{: file='reproducibility.py'}

```text
$ python reproducibility.py
[0.515164, 0.510979, 0.512556]
$ python reproducibility.py deterministic
[0.512156, 0.512156, 0.512156]
```
{: file='GPU 上的输出结果'}

| 设置                                        | 作用 |
| :------------------------------------------ | :--- |
| `torch.backends.cudnn.benchmark = False`    | 不让 cuDNN 根据输入尺寸试跑并挑选最快的卷积算法，挑选的结果每次可能不同 |
| `torch.backends.cudnn.deterministic = True` | 只使用确定性的 cuDNN 算法 |
| `torch.use_deterministic_algorithms(True)`  | 用到没有确定性实现的操作时直接报错，而不是悄悄给出不可复现的结果 |

- 不加这些设置时，3 次训练的损失在小数点后第 3 位就开始不同，每次运行脚本的结果也不一样；加上之后完全相同
- 代价是速度变慢，并且有些操作在 GPU 上没有确定性的实现。例如输出大小不为 1 的`nn.AdaptiveAvgPool2d`在反向传播时会报错`adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation`。此时可以换用其他操作，或者改为`torch.use_deterministic_algorithms(True, warn_only=True)`，只警告不报错

实际使用中不必追求逐位相同。固定种子保证结果大体可比，再用几个不同的种子各训练一次，观察指标的波动范围，比单次的结果更有说服力
