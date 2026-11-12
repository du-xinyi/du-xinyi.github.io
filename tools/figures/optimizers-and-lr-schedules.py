import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json
from lib import *
import numpy as np
T = '2026-11-12-optimizers-and-lr-schedules'
W = HERE + '/data/optimizers-and-lr-schedules/'
C = {'blue': '#2f6fd0', 'orange': '#d9822b', 'green': '#2e9950', 'purple': '#7a4fd0', 'red': '#d64545', 'grey': '#868e96'}
which = sys.argv[1:] or ['trajectories', 'adam', 'decay', 'pitfall', 'batch']

def style(ax):
    ax.grid(True, color='#e9ecef', lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#adb5bd')
    ax.tick_params(colors='#495057')

# ---- 1. 病态二次函数上的轨迹（数据来自文中代码的 paths）
if 'trajectories' in which:
    paths = json.load(open(W + 'trajectories.json'))
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    xs, ys = np.meshgrid(np.linspace(-9, 3, 300), np.linspace(-4.3, 4.3, 300))
    zs = 0.5 * (xs ** 2 + 50 * ys ** 2)
    titles = [('SGD', 'SGD，lr = 0.039', C['blue']), ('动量', '动量，lr = 0.061，momentum = 0.566', C['orange']), ('Adam', 'Adam，lr = 0.3', C['green'])]
    for ax, (key, title, c) in zip(axs, titles):
        p = np.array(paths[key])
        ax.contour(xs, ys, zs, levels=[0.5, 2, 8, 20, 40, 70, 110, 170, 250, 350], colors='#ced4da', linewidths=0.9)
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.3, marker='o', ms=3)
        ax.scatter([p[0, 0]], [p[0, 1]], color=INK, s=30, zorder=5); ax.scatter([0], [0], marker='*', color=C['red'], s=110, zorder=5)
        k = next((i for i in range(len(p)) if 0.5 * (p[i, 0] ** 2 + 50 * p[i, 1] ** 2) < 1e-2), None)
        ax.set_title(title, fontsize=11); ax.set_xlabel('x（曲率 1）')
        ax.text(0.98, 0.04, f'损失低于 0.01：第 {k} 步', transform=ax.transAxes, ha='right', fontsize=9.5, color='#495057')
        ax.set_xlim(-9, 3); ax.set_ylim(-4.3, 4.3)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    axs[0].set_ylabel('y（曲率 50）')
    fig.tight_layout(); save(fig, T, 'trajectories.png')

# ---- 2. Adam 前若干步的步长
if 'adam' in which:
    d = json.load(open(W + 'adam_steps.json')); t = np.arange(1, 1001)
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axs[0].plot(t, d['no_correction'], color=C['red'], lw=2, label='去掉偏差修正')
    axs[0].plot(t, np.ones(1000), color=C['blue'], lw=2, label='有偏差修正（torch.optim.Adam）')
    axs[0].set_title('梯度恒定时每一步的步长', fontsize=11.5); axs[0].set_ylim(0, 7.2)
    axs[0].annotate('第 12 步，6.57', (12, 6.57), (1.1, 6.5), fontsize=9.5, color=C['red'], arrowprops=dict(arrowstyle='-', color=C['red'], lw=1))
    axs[1].plot(t, d['noise'], color=C['blue'], lw=2, label='有偏差修正（torch.optim.Adam）')
    axs[1].axhline(d['noise'][-1], color=C['grey'], lw=1, ls='--')
    axs[1].text(1.15, d['noise'][-1] + 0.03, f"稳定后约 {d['noise'][-1]:.2f}", fontsize=9.5, color=C['grey'])
    axs[1].set_title('梯度为纯噪声时每一步的平均步长', fontsize=11.5); axs[1].set_ylim(0, 1.1)
    for ax in axs:
        ax.set_xscale('log'); ax.set_xlabel('步数（对数坐标）'); ax.set_ylabel('|Δw| / lr'); style(ax); ax.legend(frameon=False, fontsize=9.5, loc='upper right')
    fig.tight_layout(w_pad=2.5); save(fig, T, 'adam_steps.png')

# ---- 3. Adam 加 L2 与 AdamW 的实际衰减
if 'decay' in which:
    d = json.load(open(W + 'weight_decay.json')); t = np.arange(0, 2001, 10)
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(t, d['Adam 10.0'], color=C['orange'], lw=2, label='Adam(weight_decay=0.1)，梯度标准差 10')
    ax.plot(t, d['Adam 0.01'], color=C['red'], lw=2, label='Adam(weight_decay=0.1)，梯度标准差 0.01')
    ax.plot(t, d['AdamW 10.0'], color=C['blue'], lw=2, label='AdamW(weight_decay=0.1)，梯度标准差 10')
    ax.plot(t, d['AdamW 0.01'], color=C['green'], lw=2, ls=(0, (4, 3)), label='AdamW(weight_decay=0.1)，梯度标准差 0.01')
    ax.set_xlabel('步数'); ax.set_ylabel('权重的平均值（初值为 1）'); ax.set_ylim(0, 1.05)
    ax.set_title('同样的 weight_decay 在梯度大小不同的参数上产生的衰减', fontsize=11.5)
    style(ax); ax.legend(frameon=False, fontsize=9.5, loc='lower left', bbox_to_anchor=(0.0, 0.08)); fig.tight_layout(); save(fig, T, 'weight_decay.png')

# ---- 4. scheduler.step() 的调用频率
if 'pitfall' in which:
    d = json.load(open(W + 'step_pitfall.json')); t = np.arange(1000) / 50
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
    axs[0].plot(t, d['按步数设置，每个批次调用'], color=C['blue'], lw=2, label='T_max = 1000，每个批次调用')
    axs[0].plot(t, d['按轮数设置，每轮调用'], color=C['green'], lw=2, label='T_max = 20，每轮调用')
    axs[0].set_title('单位一致', fontsize=11.5)
    axs[1].plot(t, d['按轮数设置，每个批次调用'], color=C['red'], lw=1.6, label='T_max = 20，每个批次调用')
    axs[1].plot(t, d['按步数设置，每轮调用'], color=C['orange'], lw=2, label='T_max = 1000，每轮调用')
    axs[1].set_title('单位不一致', fontsize=11.5)
    for ax in axs:
        ax.set_xlabel('轮数（每轮 50 个批次）'); ax.set_xticks(range(0, 21, 5)); ax.set_ylim(0, 1.32e-3); style(ax)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True); ax.legend(frameon=False, fontsize=9.5, loc='upper right', ncol=2)
    axs[0].set_ylabel('学习率')
    fig.tight_layout(w_pad=2); save(fig, T, 'step_pitfall.png')

# ---- 5. 批大小与学习率
if 'batch' in which:
    d = json.load(open(W + 'batch_size.json')); ep = np.arange(1, 41)
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    cs = {'16': C['blue'], '64': C['green'], '256': C['orange'], '1024': C['red']}
    for k in ['16 0.01', '64 0.01', '256 0.01', '1024 0.01']:
        b, lr = k.split(); axs[0].plot(ep, d[k], color=cs[b], lw=2, label=f'批大小 {b}，lr = {lr}')
    for k in ['16 0.01', '64 0.04', '256 0.16', '1024 0.64']:
        b, lr = k.split(); axs[1].plot(ep, d[k], color=cs[b], lw=2, label=f'批大小 {b}，lr = {lr}')
    axs[0].set_title('学习率固定为 0.01', fontsize=11.5); axs[1].set_title('学习率随批大小等比例放大', fontsize=11.5)
    for ax in axs:
        ax.set_yscale('log'); ax.set_xlabel('轮数'); style(ax); ax.legend(frameon=False, fontsize=9.5, loc='lower left')
    axs[0].set_ylabel('训练损失（对数坐标，3 个种子平均）')
    fig.tight_layout(w_pad=2); save(fig, T, 'batch_size.png')
