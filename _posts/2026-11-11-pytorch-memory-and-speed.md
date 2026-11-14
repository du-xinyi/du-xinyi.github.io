---
title: PyTorch Memory and Speed
description: 整理 PyTorch 训练中显存不足的原因与处理方法，以及怎样找出训练速度的瓶颈
date: 2026-11-11 14:20:00 +0800
categories: [ Deep Learning, PyTorch ]
tags: [ PyTorch, Python, Deep Learning, Debugging ]
series: deep-learning
---

训练能够正常收敛之后，剩下的问题通常是显存不够用和训练太慢。这里整理显存占用的来源、显存不足时的处理方法，以及怎样找出速度的瓶颈。结果不正确时的排查见[PyTorch Debugging](/posts/pytorch-debugging/)

## 显存不足

显存主要被三部分占用：模型的参数、每一层为反向传播保存的中间结果、优化器的状态。其中中间结果与批大小成正比，通常占大部分

| 方法                        | 说明 |
| :-------------------------- | :--- |
| 减小批大小                  | 最直接的方法，配合梯度累积可以保持等效的批大小不变 |
| 验证时使用`torch.no_grad()` | 不保存中间结果 |
| 累加损失时使用`loss.item()` | 直接累加`loss`会把每一步的计算图都保留下来，显存随步数不断增长 |
| 减小输入尺寸或模型          | 卷积网络的中间结果与输入的面积成正比 |

下面的程序需要 GPU，输出来自一台 CPU 为 Intel Core Ultra 9 285K、GPU 为 GeForce RTX 5090 D（32 GB 显存）、内存为 64 GB 的机器

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1)).to(device)
x = torch.randn(256, 1024, device=device)

def mb(n):
    return f"{n / 1024 ** 2:.0f} MB"

if device == "cuda":
    # 累加损失时不加item()，每一步的计算图都留在显存里
    for use_item in (False, True):
        start = torch.cuda.memory_allocated()
        total = 0.0
        for step in range(200):
            loss = model(x).pow(2).mean()
            total += loss.item() if use_item else loss
        print(f"item()={use_item}  显存增加 {mb(torch.cuda.memory_allocated() - start)}")
        del total, loss

    # 验证时不加no_grad，中间结果会为了反向传播而保留
    big = torch.randn(8192, 1024, device=device)
    for no_grad in (False, True):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        with torch.set_grad_enabled(not no_grad):
            out = model(big)
        print(f"no_grad={no_grad}  前向传播后占用 {mb(torch.cuda.memory_allocated() - start)}"
              f"  过程中的峰值 {mb(torch.cuda.max_memory_allocated() - start)}")
        del out

    # 显存不足时抛出torch.OutOfMemoryError，可以捕获后减小批大小重试
    try:
        torch.empty(1024, 1024, 1024, 64, device=device)  # 256 GB
    except torch.OutOfMemoryError as e:
        print(str(e).split(".")[0])

    print(f"已分配 {mb(torch.cuda.memory_allocated())}  已缓存 {mb(torch.cuda.memory_reserved())}  峰值 {mb(torch.cuda.max_memory_allocated())}")
    del big
    torch.cuda.empty_cache()  # 把没有使用的缓存还给系统，nvidia-smi中的占用才会下降
    print(f"已分配 {mb(torch.cuda.memory_allocated())}  已缓存 {mb(torch.cuda.memory_reserved())}")
```
{: file='查看显存的占用'}

```text
item()=False  显存增加 200 MB
item()=True  显存增加 1 MB
no_grad=False  前向传播后占用 32 MB  过程中的峰值 64 MB
no_grad=True  前向传播后占用 0 MB  过程中的峰值 64 MB
CUDA out of memory
已分配 37 MB  已缓存 54 MB  峰值 101 MB
已分配 5 MB  已缓存 22 MB
```
{: file='输出结果'}

- 累加`loss`而不是`loss.item()`时，200 步之后多占用了 200 MB，并且会一直增长下去。训练若干轮之后才出现的显存不足基本都是这个原因，把张量存进列表（例如保存每个批次的输出用于统计）也是一样，要先`detach()`再保存
- 不加`no_grad`时，前向传播结束后中间结果仍然占着显存；层数越多，保留下来的越多
- 显存不足时抛出`torch.OutOfMemoryError`，报错信息中包含要申请的大小和当前的占用
- `memory_allocated`是张量实际占用的显存，`memory_reserved`是 PyTorch 向系统申请后缓存起来的显存，`nvidia-smi`看到的是后者再加上 CUDA 自身的开销。`empty_cache()`只归还没有被使用的缓存，不会释放仍被变量引用的张量

显存放不下需要的批大小时，可以使用梯度累积：连续计算几个小批次的梯度，累加之后再更新一次参数，效果等同于一个大批次。写法和需要注意的地方见[PyTorch Training](/posts/pytorch-training/)中的梯度累积一节。批大小本身也放不下时，可以把训练分到多张显卡上，见[PyTorch Distributed Training](/posts/pytorch-distributed-training/)

## 找出速度的瓶颈

训练慢时先分清时间花在哪里：数据读取在 CPU 上进行，模型的计算在 GPU 上进行，两者的优化方法完全不同。在训练循环中分别统计等待数据的时间和模型计算的时间即可

```python
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class SlowDataset(Dataset):
    def __len__(self):
        return 512

    def __getitem__(self, index):
        time.sleep(0.002)  # 模拟读取文件和数据增强的耗时
        return torch.randn(3, 32, 32), index % 10

def profile(loader, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data_time = model_time = 0.0
    end = time.perf_counter()
    for x, y in loader:
        start = time.perf_counter()
        data_time += start - end  # 等待数据的时间
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.perf_counter()
        model_time += end - start  # 前向、反向和更新参数的时间
    return data_time, model_time

if __name__ == "__main__":
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
    for num_workers in (0, 4):
        loader = DataLoader(SlowDataset(), batch_size=64, num_workers=num_workers)
        data_time, model_time = profile(loader, model)
        print(f"num_workers={num_workers}  数据 {data_time:.2f} s  模型 {model_time:.2f} s  数据占比 {data_time / (data_time + model_time):.0%}")
```
{: file='分别统计数据读取与模型计算的时间'}

```text
num_workers=0  数据 1.08 s  模型 0.41 s  数据占比 73%
num_workers=4  数据 0.20 s  模型 0.20 s  数据占比 51%
```
{: file='输出结果'}

| 瓶颈     | 表现                                           | 处理方法 |
| :------- | :--------------------------------------------- | :--- |
| 数据读取 | 等待数据的时间占大部分，GPU 利用率低且忽高忽低 | 增大`num_workers`，把解码和缩放等耗时的操作提前做好，把数据放到固态硬盘 |
| 模型计算 | 模型计算的时间占大部分，GPU 利用率接近 100%    | 混合精度，减小模型或输入尺寸，使用更大的批大小 |

- 使用多进程读取数据时，主程序要放在`if __name__ == "__main__":`之下
- `num_workers`并不是越大越好，超过 CPU 核数之后反而变慢，从 4 或 8 开始尝试
- 这里`num_workers`从 0 增加到 4 之后，等待数据的时间从 1.08 秒降到 0.20 秒，数据的占比从 73% 降到 51%
- 耗时每次运行都会有波动，上面的数值只用来比较相对大小

在 GPU 上计时还有一点要注意：GPU 上的运算是异步的，函数返回时计算还没有完成。计时结束之前要调用`torch.cuda.synchronize()`等待计算完成，否则测到的只是提交任务的时间

```python
import time

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 10)).to(device)
x = torch.randn(4096, 2048, device=device)

with torch.no_grad():
    for _ in range(5):
        model(x)  # 预热，前几次运行包含初始化的开销

    for sync in (False, True):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            out = model(x)
        if sync and device == "cuda":
            torch.cuda.synchronize()  # 等待GPU上的计算全部完成
        print(f"synchronize={sync}  {(time.perf_counter() - start) * 1000:.1f} ms")
        if device == "cuda":
            torch.cuda.synchronize()
```
{: file='在 GPU 上计时'}

```text
synchronize=False  0.6 ms
synchronize=True  27.4 ms
```
{: file='输出结果'}

不调用`synchronize`时测到的时间只有 0.6 毫秒，实际的计算用了 27.4 毫秒。上一段统计模型计算时间的代码在 GPU 上使用时，同样要在读取时间之前加上`torch.cuda.synchronize()`；`loss.item()`、`print`、`.cpu()`等需要取回结果的操作也会等待计算完成，耗时会被算在这些操作上
