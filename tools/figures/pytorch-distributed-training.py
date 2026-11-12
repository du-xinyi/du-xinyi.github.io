import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys
from lib import *
import json
import numpy as np
W = HERE + '/data/pytorch-distributed-training/'
T = '2026-11-12-pytorch-distributed-training'

# ---- 1. how DDP works
fig, ax = canvas(15, 8.6, 0.78)
box(ax, 5.0, 7.7, 5.0, 0.7, '训练集', GREY, fs=11, bold=True)
box(ax, 4.0, 6.5, 7.0, 0.7, 'DistributedSampler：每一轮打乱后按 rank 切分', GREY, fs=10)
arrow(ax, (7.5, 7.7), (7.5, 7.2))
for i, x in enumerate([0.4, 9.6]):
    c = BLUE if i == 0 else GREEN
    box(ax, x, 0.3, 5.0, 5.4, '', (c[0] + '55', c[1]), lw=1.2, ls='--')
    label(ax, x + 2.5, 5.38, f'进程 rank {i}（显卡 {i}）', 11, c[1], bold=True)
    arrow(ax, (7.5 + (-1.6 if i == 0 else 1.6), 6.5), (x + 2.5, 5.7), color=c[1])
    box(ax, x + 0.7, 4.3, 3.6, 0.62, f'第 {i + 1} 份数据的一个批次', c, fs=9.8)
    box(ax, x + 0.7, 3.25, 3.6, 0.62, '模型副本：前向传播、计算损失', c, fs=9.8)
    box(ax, x + 0.7, 2.2, 3.6, 0.62, '反向传播，得到本进程的梯度', c, fs=9.8)
    box(ax, x + 0.7, 0.6, 3.6, 0.62, '优化器更新本进程的参数', c, fs=9.8)
    arrow(ax, (x + 2.5, 4.3), (x + 2.5, 3.87)); arrow(ax, (x + 2.5, 3.25), (x + 2.5, 2.82))
box(ax, 6.0, 1.35, 3.0, 0.8, '所有进程的梯度\n取平均（all-reduce）', ORANGE, fs=9.6)
arrow(ax, (4.7, 2.45), (6.0, 1.95), color=ORANGE[1]); arrow(ax, (10.3, 2.45), (9.0, 1.95), color=ORANGE[1])
arrow(ax, (6.0, 1.55), (4.7, 1.0), color=ORANGE[1]); arrow(ax, (9.0, 1.55), (10.3, 1.0), color=ORANGE[1])
label(ax, 7.5, 2.5, '各进程的梯度', 9, ORANGE[1]); label(ax, 7.5, 0.75, '平均后的梯度', 9, ORANGE[1])
label(ax, 7.5, 4.6, '初始参数相同、每一步的梯度相同\n各进程的参数始终相同', 9.2, '#868e96')
save(fig, T, 'ddp.png')

# ---- 2. DataParallel
fig, ax = canvas(15, 6.2, 0.78)
box(ax, 0.4, 0.3, 14.2, 5.6, '', (GREY[0] + '66', GREY[1]), lw=1.2, ls='--')
label(ax, 7.5, 5.55, '一个进程', 11, GREY[1], bold=True)
box(ax, 0.9, 2.6, 2.6, 0.8, '一个批次\n（在显卡 0 上）', GREY, fs=9.6)
box(ax, 4.6, 2.6, 1.9, 0.8, '拆分数据\n复制模型', ORANGE, fs=9.6)
arrow(ax, (3.5, 3.0), (4.6, 3.0))
for i, y in enumerate([4.0, 2.6, 1.2]):
    c = [BLUE, GREEN, PURPLE][i]
    box(ax, 7.6, y, 3.0, 0.8, f'显卡 {i}：模型副本\n前向传播', c, fs=9.6)
    arrow(ax, (6.5, 3.0), (7.6, y + 0.4), color=c[1]); arrow(ax, (10.6, y + 0.4), (11.7, 3.0), color=c[1])
box(ax, 11.7, 2.6, 2.5, 0.8, '输出汇总到显卡 0\n计算损失', ORANGE, fs=9.6)
ax.add_patch(FancyArrowPatch((12.95, 2.6), (5.55, 2.6), arrowstyle='-|>', mutation_scale=13, color='#868e96', lw=1.2, ls='--', connectionstyle='arc3,rad=-0.42'))
label(ax, 5.2, 0.7, '反向传播的梯度汇总到显卡 0 上的模型\n每个批次都重复一遍', 9.2, '#868e96')
save(fig, T, 'data_parallel.png')

# ---- 3. DistributedSampler (indices are the real output of sampler.py)
fig, ax = canvas(15, 6.6, 0.78)
def cells(x, y, vals, c, w=0.62, fs=9.5, hl=()):
    for i, v in enumerate(vals):
        box(ax, x + i * w, y, w - 0.06, 0.52, str(v), RED if i in hl else c, fs=fs, round=0.05, lw=1.1)
label(ax, 3.6, 6.25, '训练：打乱后按 rank 间隔取出', 11.5, bold=True)
label(ax, 0.3, 5.45, '下标', 9.5, ha='left'); cells(1.0, 5.2, range(10), GREY)
arrow(ax, (4.1, 5.2), (4.1, 4.6), '按 seed + epoch 打乱', off=(1.35, 0.0))
perm = [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]
for i, v in enumerate(perm): box(ax, 1.0 + i * 0.62, 4.05, 0.56, 0.52, str(v), BLUE if i % 2 == 0 else GREEN, fs=9.5, round=0.05, lw=1.1)
arrow(ax, (3.0, 4.05), (2.3, 3.2), color=BLUE[1]); arrow(ax, (5.2, 4.05), (5.6, 3.2), color=GREEN[1])
label(ax, 0.3, 2.9, 'rank 0', 9.5, BLUE[1], ha='left', bold=True); cells(1.3, 2.65, [4, 7, 3, 0, 6], BLUE)
label(ax, 4.75, 2.9, 'rank 1', 9.5, GREEN[1], ha='left', bold=True); cells(5.75, 2.65, [1, 5, 9, 8, 2], GREEN)
label(ax, 3.9, 1.6, '两个进程取到的样本互不重复\n调用 set_epoch 后每一轮的划分都不同', 9.3, '#868e96')
ax.plot([9.3, 9.3], [0.3, 6.3], color='#dee2e6', lw=1)
label(ax, 12.2, 6.25, '10 个样本分给 4 个进程：补齐', 11.5, bold=True)
cells(9.8, 5.2, list(range(10)) + [0, 1], GREY, w=0.42, fs=8.8, hl=(10, 11))
label(ax, 12.3, 4.75, '末尾重复开头的 2 个样本，凑成 12 个', 9, RED[1])
for r, vals in enumerate([[0, 4, 8], [1, 5, 9], [2, 6, 0], [3, 7, 1]]):
    y = 3.7 - r * 0.72
    label(ax, 10.0, y + 0.26, f'rank {r}', 9.5, ha='left', bold=True)
    cells(11.3, y, vals, [BLUE, GREEN, ORANGE, PURPLE][r], hl=(2,) if r >= 2 else ())
label(ax, 12.2, 0.75, '样本 0 和 1 各被评估了两次', 9.3, '#868e96')
save(fig, T, 'sampler.png')

# ---- 4. equivalence curves
d = json.load(open(W + 'curves.json'))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axs[0]
a, b = np.array(d['2x32_lr0.1']), np.array(d['1x64_lr0.1'])
ax.plot(np.arange(len(b)) + 1, b, color='#adb5bd', lw=5, label='1 个进程，批大小 64')
ax.plot(np.arange(len(a)) + 1, a, color=BLUE[1], lw=1.3, label='2 个进程，每个进程批大小 32')
ax.set_xlabel('步数'); ax.set_ylabel('训练损失'); ax.set_title('两条曲线重合：每一步损失的最大差值为 4.77e-07', fontsize=11)
ax.legend(frameon=False); ax.grid(alpha=0.3)
ax = axs[1]
def smooth(v, k=9):
    v = np.array(v); return np.convolve(np.pad(v, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid')
for key, name, c, ls in [('1x32_lr0.1', '1 个进程 × 32，学习率 0.1', GREY[1], '-'), ('2x32_lr0.1', '2 个进程 × 32，学习率 0.1', BLUE[1], '-'),
                         ('2x32_lr0.2', '2 个进程 × 32，学习率 0.2', ORANGE[1], '-')]:
    v = smooth(d[key]); ax.plot(np.arange(1, len(v) + 1) / len(v) * 5, v, color=c, ls=ls, lw=1.6, label=name)
ax.set_yscale('log'); ax.set_xlabel('轮数'); ax.set_ylabel('训练损失（9 步滑动平均）'); ax.set_title('同样的轮数，2 个进程的更新步数只有一半', fontsize=11)
ax.legend(frameon=False); ax.grid(alpha=0.3, which='both')
fig.tight_layout(w_pad=2.5); save(fig, T, 'equivalence.png')
