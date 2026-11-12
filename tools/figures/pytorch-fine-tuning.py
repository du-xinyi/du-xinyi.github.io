import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json, os
from lib import *
import numpy as np
T = '2026-11-11-pytorch-fine-tuning'
W = HERE + '/data/pytorch-fine-tuning/'
which = sys.argv[1:] or ['strategies', 'lora', 'schedule', 'curves']

# ---- 1. strategies
if 'strategies' in which:
    fig, ax = canvas(16.4, 7.4, 0.76)
    cols = [('从头训练', [GREEN, GREEN, GREEN], False), ('特征提取', [GREY, GREY, GREEN], False),
            ('全量微调', [ORANGE, ORANGE, GREEN], False), ('参数高效微调', [GREY, GREY, GREEN], True)]
    names = ['主干前部\n（边缘、纹理等通用特征）', '主干后部\n（与任务相关的特征）', '分类头']
    hs = [1.5, 1.5, 0.7]
    for i, (title, cs, peft) in enumerate(cols):
        x = 0.5 + i * 4.0; w = 2.9
        label(ax, x + w / 2, 7.0, title, 12, bold=True)
        box(ax, x, 5.75, w, 0.55, '输入图像', BLUE, fs=9.4)
        y = 5.75; prev = y
        for name, c, h in zip(names, cs, hs):
            y2 = prev - 0.35 - h
            box(ax, x, y2, w, h, name, c, fs=9.2)
            arrow(ax, (x + w / 2, prev), (x + w / 2, y2 + h), lw=1.2)
            if peft and h > 1:
                box(ax, x + w + 0.12, y2 + 0.3, 0.62, h - 0.6, '适\n配\n器', PURPLE, fs=8.2)
            prev = y2
        box(ax, x, prev - 0.35 - 0.5, w, 0.5, '输出', BLUE, fs=9.4); arrow(ax, (x + w / 2, prev), (x + w / 2, prev - 0.35), lw=1.2)
    ly = -0.55; items = [(GREEN, '随机初始化，参与训练'), (ORANGE, '预训练权重，参与训练'), (GREY, '预训练权重，冻结'), (PURPLE, '新增的少量参数，参与训练')]
    lx = 0.9
    for c, t in items:
        box(ax, lx, ly - 0.17, 0.5, 0.34, '', c); label(ax, lx + 0.65, ly, t, 9.2, ha='left'); lx += 3.9
    ax.set_ylim(-1.0, 7.4); fig.set_size_inches(16.4 * 0.76, 8.4 * 0.76)
    save(fig, T, 'strategies.png')

# ---- 2. lora
if 'lora' in which:
    fig, ax = canvas(15.2, 6.2, 0.8)
    label(ax, 4.2, 5.85, '训练时：原权重冻结，只训练 A 和 B', 12, bold=True)
    box(ax, 3.3, 0.3, 1.8, 0.55, '输入 x', BLUE, fs=9.6)
    box(ax, 0.6, 1.9, 3.0, 2.0, '原权重 $W_0$\n$d \\times k$\n冻结', GREY, fs=10)
    ax.add_patch(plt.Polygon([[5.0, 1.75], [7.8, 1.75], [7.1, 2.55], [5.7, 2.55]], fc=ORANGE[0], ec=ORANGE[1], lw=1.4))
    label(ax, 6.4, 2.15, '$A$：$r \\times k$', 10)
    ax.add_patch(plt.Polygon([[5.7, 3.2], [7.1, 3.2], [7.8, 4.0], [5.0, 4.0]], fc=ORANGE[0], ec=ORANGE[1], lw=1.4))
    label(ax, 6.4, 3.6, '$B$：$d \\times r$', 10)
    arrow(ax, (6.4, 2.55), (6.4, 3.2), lw=1.2); label(ax, 6.55, 2.875, '维度 $r$', 8.6, '#868e96', ha='left')
    label(ax, 8.0, 2.15, '随机初始化', 8.6, '#868e96', ha='left'); label(ax, 8.0, 3.6, '初始为 0', 8.6, '#868e96', ha='left')
    arrow(ax, (3.9, 0.85), (2.1, 1.9), lw=1.2); arrow(ax, (4.5, 0.85), (6.4, 1.75), lw=1.2)
    ax.add_patch(Circle((4.2, 4.75), 0.26, fc='white', ec=INK, lw=1.3)); label(ax, 4.2, 4.75, '+', 13)
    arrow(ax, (2.1, 3.9), (3.98, 4.62), lw=1.2); arrow(ax, (6.4, 4.0), (4.42, 4.62), lw=1.2)
    label(ax, 5.75, 4.5, '$\\times\\ \\alpha / r$', 9.2, '#d9822b', ha='left')
    arrow(ax, (4.2, 5.01), (4.2, 5.45), lw=1.2); label(ax, 4.5, 5.3, '输出 h', 9.6, ha='left')
    ax.plot([9.9, 9.9], [0.2, 5.9], color='#dee2e6', lw=1)
    label(ax, 12.5, 5.85, '推理时：合并为一个权重', 12, bold=True)
    box(ax, 11.6, 0.3, 1.8, 0.55, '输入 x', BLUE, fs=9.6)
    box(ax, 10.6, 1.9, 3.8, 2.0, '$W = W_0 + (\\alpha / r) \\cdot BA$\n$d \\times k$', GREEN, fs=10.5)
    arrow(ax, (12.5, 0.85), (12.5, 1.9), lw=1.2); arrow(ax, (12.5, 3.9), (12.5, 4.5), lw=1.2); label(ax, 12.5, 4.75, '输出 h', 9.6)
    label(ax, 12.5, 5.35, '结构与原模型完全相同，没有额外开销', 8.8, '#868e96')
    save(fig, T, 'lora.png')

# ---- 3. schedule
if 'schedule' in which:
    fig, ax = canvas(13.6, 6.6, 0.8)
    rows = ['fc（新的分类头）', 'layer4', 'layer3', 'layer2', 'layer1', 'conv1、bn1']
    stages = ['第 1 阶段', '第 2 阶段', '第 3 阶段']
    lrs = [['1e-3', '1e-3', '1e-3'], [None, '1e-4', '1e-4'], [None, None, '3e-5'], [None, None, None], [None, None, None], [None, None, None]]
    x0, cw, rh = 3.6, 2.9, 0.72
    for j, s in enumerate(stages): label(ax, x0 + j * (cw + 0.25) + cw / 2, 6.0, s, 11.5, bold=True)
    for i, r in enumerate(rows):
        y = 5.0 - i * (rh + 0.16)
        label(ax, x0 - 0.3, y + rh / 2, r, 10, ha='right', mono=False)
        for j in range(3):
            v = lrs[i][j]; x = x0 + j * (cw + 0.25)
            c = GREY if v is None else (GREEN if i == 0 else ORANGE)
            box(ax, x, y, cw, rh, '冻结' if v is None else f'训练，学习率 {v}', c, fs=9.4, tc='#868e96' if v is None else INK)
    arrow(ax, (0.7, 0.75), (0.7, 5.6), lw=1.3); label(ax, 0.45, 5.85, '靠近输出', 9.2, '#868e96', ha='left'); label(ax, 0.45, 0.5, '靠近输入', 9.2, '#868e96', ha='left')
    label(ax, x0 + 1.5 * cw + 0.25, 0.05, '从靠近输出的层开始逐步解冻，越靠近输入学习率越小', 9.4, '#868e96')
    ax.set_ylim(-0.3, 6.6); fig.set_size_inches(13.6 * 0.8, 6.9 * 0.8)
    save(fig, T, 'schedule.png')

# ---- 4. curves
if 'curves' in which:
    res = json.load(open(W + 'curves.json'))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    sty = { 'scratch': ('从头训练', '#868e96', 's'), 'frozen': ('特征提取（冻结主干）', '#2f6fd0', 'o'), 'finetune': ('全量微调', '#d9822b', '^') }
    for k, (name, c, m) in sty.items():
        ax.plot(range(len(res[k])), res[k], color=c, marker=m, ms=5, lw=1.8, label=name)
    ax.set_xlabel('训练轮数'); ax.set_ylabel('验证集准确率'); ax.set_ylim(0, 1.03); ax.grid(alpha=0.3)
    ax.set_title('ResNet18 在 300 张手写数字上的三种训练方式', fontsize=11.5)
    ax.legend(loc='lower right', frameon=False)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout(); save(fig, T, 'curves.png')

# ---- 5. pitfalls
if 'pitfalls' in which or len(sys.argv) == 1:
    res = json.load(open(W + 'curves.json')); pit = json.load(open(W + 'pitfalls.json'))
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    first = res['finetune'][0]
    panels = [('全量微调：学习率的影响', [('主干 1e-4，分类头 1e-3', res['finetune'], '#d9822b', '^'), ('全部参数 1e-2', [first] + pit['finetune_lr1e-2'], '#d64545', 'x')]),
              ('冻结主干：BatchNorm 模式的影响', [('BatchNorm 保持 eval()', res['frozen'], '#2f6fd0', 'o'), ('BatchNorm 处于 train()', [first] + pit['frozen_bn_train'], '#d64545', 'x')])]
    for ax, (title, lines) in zip(axs, panels):
        for name, ys, c, m in lines: ax.plot(range(len(ys)), ys, color=c, marker=m, ms=5, lw=1.8, label=name)
        ax.set_title(title, fontsize=11.5); ax.set_xlabel('训练轮数'); ax.set_ylim(0, 1.03); ax.grid(alpha=0.3); ax.legend(loc='upper left' if ax is axs[0] else 'lower right', frameon=False)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    axs[0].set_ylabel('验证集准确率')
    fig.tight_layout(); save(fig, T, 'pitfalls.png')
