---
title: TensorBoard
description: 整理 TensorBoard 在 PyTorch 中的用法，包括 SummaryWriter 的各种记录方法、启动与远程查看，以及记录频率、关闭写入器、事件文件大小等常见问题
date: 2026-11-12 09:20:00 +0800
categories: [ Deep Learning, PyTorch ]
tags: [ PyTorch, Python, Deep Learning, TensorBoard ]
series: deep-learning
---

TensorBoard 读取训练时写下的事件文件，在浏览器中画出曲线、直方图和图片。PyTorch 通过`torch.utils.tensorboard.SummaryWriter`写事件文件，需要另外安装`tensorboard`包。怎样在训练脚本中组织配置、日志和指标，见[Experiment Tracking](/posts/experiment-tracking/)

## SummaryWriter

```python
import math
import os

import torch
from sklearn.datasets import load_digits
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

torch.manual_seed(0)
writer = SummaryWriter("tb_demo/run1")  # 不传目录时写入 runs/<月日_时-分-秒>_<主机名>

for step in range(100):
    train_loss, val_loss = math.exp(-step / 30), math.exp(-step / 30) + 0.1
    writer.add_scalar("loss/train", train_loss, step)  # 标签中斜杠之前的部分是分组名
    writer.add_scalar("loss/val", val_loss, step)
    writer.add_scalars("loss_both", {"train": train_loss, "val": val_loss}, step)  # 画在同一张图里

for epoch in range(3):
    writer.add_histogram("weights/fc", torch.randn(1000) * (epoch + 1), epoch)     # 任意形状的张量

images = torch.from_numpy(load_digits().images[:32]).float().unsqueeze(1) / 16     # (32, 1, 8, 8)，取值0到1
grid = make_grid(images, nrow=8, padding=1, pad_value=1.0)                         # 拼成一张大图
print(grid.shape)                                                                  # torch.Size([3, 37, 73])
writer.add_image("inputs", grid, 0)                                                # 默认要求 (C, H, W)

writer.add_text("config", "lr = 0.003, batch_size = 64", 0)                        # 支持Markdown
writer.add_hparams({"lr": 3e-3, "batch_size": 64}, {"hparam/accuracy": 0.99})      # 超参数与最终指标
writer.close()

for root, _, files in sorted(os.walk("tb_demo")):
    print(root, len(files))
```
{: file='SummaryWriter 的常用方法'}

```text
torch.Size([3, 37, 73])
tb_demo 0
tb_demo/run1 1
tb_demo/run1/1794457154.4981537 1
tb_demo/run1/loss_both_train 1
tb_demo/run1/loss_both_val 1
```
{: file='输出结果'}

| 方法                                    | 记录的内容           | 说明 |
| :-------------------------------------- | :------------------- | :--- |
| `add_scalar(tag, value, step)`          | 一个数值             | 损失、准确率、学习率。`value`可以是 Python 数或单元素张量 |
| `add_scalars(main_tag, dict, step)`     | 多个数值画在一张图里 | 每个键单独建一个子目录，在界面中显示为单独的运行 |
| `add_histogram(tag, tensor, step)`      | 张量中数值的分布     | 观察权重、梯度、激活值随训练的变化 |
| `add_image(tag, tensor, step)`          | 一张图片             | 默认形状`(C, H, W)`，浮点数取值 0 到 1；多张图先用`make_grid`拼起来 |
| `add_text(tag, text, step)`             | 一段文字             | 支持 Markdown，适合记录配置 |
| `add_hparams(hparam_dict, metric_dict)` | 超参数和最终指标     | 在 HPARAMS 页面中把多次实验列成表格 |
| `flush()`、`close()`                    | 把缓冲区写入磁盘     | 训练结束时必须`close()` |

| 构造参数     | 默认值 | 作用                                                     |
| :----------- | :----- | :------------------------------------------------------- |
| `log_dir`    | `None` | 事件文件所在的目录，不传时自动生成`runs/<时间>_<主机名>` |
| `flush_secs` | 120    | 后台线程每隔多少秒把缓冲区写入磁盘                       |
| `max_queue`  | 10     | 缓冲队列的长度                                           |
| `purge_step` | `None` | 从检查点恢复训练时，丢弃这一步及之后的旧记录             |

- 标签中斜杠前面的部分是分组名，`loss/train`和`loss/val`在界面上归入同一个`loss`分组，但仍是两张图
- 横轴`step`要用全局步数，而不是每一轮里重新从 0 开始的批次序号，否则不同轮的点会叠在一起
- 从输出可以看到，`add_scalars`为每个键建了一个子目录，`add_hparams`又建了一个以时间戳命名的子目录。实验多了之后运行列表会很乱，所以[Experiment Tracking](/posts/experiment-tracking/)中的训练脚本只用`add_scalar`，并给`add_hparams`传`run_name="."`，让它写进实验目录本身

## 启动与远程查看

```bash
tensorboard --logdir runs --port 6006
```
{: file='启动 TensorBoard'}

然后在浏览器中打开`http://localhost:6006`。`--logdir`指向所有实验目录的上一级，每个子目录显示为一个运行（run），可以在左侧勾选要显示哪些。TensorBoard 每隔几秒重新读取一次目录，训练过程中刷新页面就能看到新的点

| 参数                | 默认值 | 作用                                         |
| :------------------ | :----- | :------------------------------------------- |
| `--logdir`          | 无     | 事件文件所在的目录，会递归查找子目录         |
| `--port`            | 6006   | 端口，被占用时换一个                         |
| `--bind_all`        | 关闭   | 监听所有网卡，局域网内的其他机器可以直接访问 |
| `--reload_interval` | 5      | 重新读取目录的间隔，单位为秒                 |

训练通常在远程的服务器上进行。TensorBoard 默认只监听`localhost`，从自己的电脑上访问时，用 SSH 端口转发把本机的端口接到服务器上，比`--bind_all`把端口暴露给整个网络更安全。ssh 的配置见[Linux Tools](/posts/linux-tools/)

```bash
# 在自己的电脑上运行：本机的16006端口转发到服务器上的localhost:6006
ssh -N -L 16006:localhost:6006 user@server
```
{: file='SSH 端口转发'}

然后在自己的电脑上打开`http://localhost:16006`。`-L`的格式是`本机端口:目标主机:目标端口`，目标主机是从服务器的角度看的，所以写`localhost`；`-N`表示只转发、不打开远程的终端，按`Ctrl+C`结束转发

## 常见问题

下面的耗时来自一台 CPU 为 Intel Core Ultra 9 285K、GPU 为 GeForce RTX 5090 D（32 GB 显存）、内存为 64 GB 的机器

### 记录太频繁

```python
import os
import shutil
import time

import torch
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
weights = torch.randn(64, 32, 3, 3)                 # 18432个数，相当于一个卷积层的权重
image = torch.rand(3, 64, 64)

def measure(name, count, log):
    shutil.rmtree(f"tb_cost/{name}", ignore_errors=True)
    writer = SummaryWriter(f"tb_cost/{name}")
    start = time.perf_counter()
    for step in range(count):
        log(writer, step)
    elapsed = time.perf_counter() - start
    writer.close()
    size = sum(os.path.getsize(f"tb_cost/{name}/{file}") for file in os.listdir(f"tb_cost/{name}"))
    print(f"{name:<18}{elapsed / count * 1e6:>9.1f} us/call{size / count:>10.0f} bytes/call")

measure("add_scalar", 20000, lambda writer, step: writer.add_scalar("loss", 0.5, step))
measure("add_scalar+flush", 2000, lambda writer, step: (writer.add_scalar("loss", 0.5, step), writer.flush()))
measure("add_histogram", 2000, lambda writer, step: writer.add_histogram("weights", weights, step))
measure("add_image", 2000, lambda writer, step: writer.add_image("image", image, step))
```
{: file='粗略测量每次调用的开销'}

```text
add_scalar             14.5 us/call        43 bytes/call
add_scalar+flush       22.0 us/call        43 bytes/call
add_histogram        1159.6 us/call      9921 bytes/call
add_image            1246.2 us/call     12473 bytes/call
```
{: file='输出结果'}

这台机器同时在运行其他任务，时间只是一个大概的量级，多次运行在 14 到 23 微秒（`add_scalar`）、1.0 到 1.2 毫秒（`add_histogram`）、1.2 到 1.4 毫秒（`add_image`）之间变化；文件大小则每次都相同

- `add_scalar`本身只要几十微秒，每一步记录一次也不会成为瓶颈。真正的开销在于取出数值：GPU 上`.item()`要等前面所有的计算完成，打断 CPU 和 GPU 的并行
- `add_histogram`和`add_image`每次约 1 毫秒，还要先把张量拷回 CPU。给模型的每一层每一步都记一次直方图，开销会超过小模型训练一步的时间。直方图、图片每几轮记录一次就够了
- `flush()`只在需要立即看到结果时调用，不要每一步都调用

### 没有关闭 SummaryWriter

`SummaryWriter`把数据放进队列，由后台线程写入文件，默认每 120 秒才把缓冲区写入磁盘一次。下面在三个子进程中分别以三种方式结束

```python
import subprocess
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 另起一个进程写入1000个点，分别以三种方式结束：正常close()、抛出异常、被强制终止
script = """
import os, sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(sys.argv[1])
for step in range(1000):
    writer.add_scalar("x", step, step)
if sys.argv[2] == "close":
    writer.close()
elif sys.argv[2] == "kill":
    os._exit(1)                                     # 相当于进程被kill -9
else:
    raise RuntimeError("训练中途出错")
"""
for mode in ("close", "error", "kill"):
    subprocess.run([sys.executable, "-c", script, f"tb_flush/{mode}", mode], stderr=subprocess.DEVNULL)
    accumulator = EventAccumulator(f"tb_flush/{mode}")
    accumulator.Reload()
    events = accumulator.Scalars("x")
    print(f"{mode:<6}{len(events):>5}  last step {events[-1].step}")
```
{: file='close_writer.py'}

```text
close  1000  last step 999
error   994  last step 993
kill    994  last step 993
```
{: file='输出结果'}

- 正常`close()`时 1000 个点都在。抛出异常、进程被强制终止时，还在队列中没有写入的最后十个左右的点丢了，具体丢几个每次运行都不一样
- 训练脚本中把`writer.close()`放进`try ... finally`，或者用`with SummaryWriter(...) as writer:`，异常退出时也能写完
- 被`kill -9`或断电时无论怎样都会丢掉最后一部分，所以重要的结果另外写进`result.json`

### 记录张量

```python
import json

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
loss = nn.Linear(4, 1)(torch.randn(8, 4)).pow(2).mean()
print(loss.requires_grad)                           # True，这个张量连着整张计算图

writer = SummaryWriter("tb_tensor")
writer.add_scalar("loss", loss, 0)                  # 可以直接传张量，内部会先detach再转成数值
writer.close()

try:
    json.dumps({"loss": loss})                      # 自己写文件时不会自动转换
except TypeError as error:
    print(error)
print(json.dumps({"loss": loss.item()}))
```
{: file='log_tensor.py'}

```text
True
Object of type Tensor is not JSON serializable
{"loss": 0.3065558969974518}
```
{: file='输出结果'}

`add_scalar`接受张量并在内部取出数值，但自己写的记录代码中，如果把`loss`张量本身存进列表留到最后再处理，每一步的计算图都会被保留下来，内存随步数不断增长。存进列表、字典、文件之前一律先`.item()`，原因见[PyTorch Debugging](/posts/pytorch-debugging/)中计算图的部分

### 事件文件过大

从上面测量的文件大小来看，一个标量约 43 字节，一个卷积层权重（18432 个数）的直方图约 9.9 KB，一张 64×64 的彩色图片约 12.5 KB。假设模型有 20 层、每层的直方图都这么大，按训练 10 万步估算：

| 记录方式                          | 每步写入        | 10 万步的大小 |
| :-------------------------------- | :-------------- | :------------ |
| 5 个标量，每步一次                | 约 215 字节     | 约 21 MB      |
| 20 层的权重直方图，每步一次       | 约 200 KB       | 约 20 GB      |
| 20 层的权重直方图，每 1000 步一次 | 平均约 200 字节 | 约 20 MB      |
| 8 张图片，每步一次                | 约 100 KB       | 约 10 GB      |

事件文件过大时，TensorBoard 启动和刷新都会很慢，还会占满硬盘。直方图和图片按轮或者每几千步记录一次，标量可以每步或每几十步记录一次

两次实验写进同一个目录、从检查点恢复训练时旧记录的处理，见[Experiment Tracking](/posts/experiment-tracking/)中的常见问题
