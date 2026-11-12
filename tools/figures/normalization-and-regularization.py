import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
from matplotlib.patches import Polygon
T = '2026-11-11-normalization-and-regularization'
W = HERE + '/data/normalization-and-regularization/'
which = sys.argv[1:] or ['dims', 'momentum', 'batch', 'overfit', 'dropout', 'compare']

# ---- 1. statistics dimensions
if 'dims' in which:
    NN, CC, HH = 5, 6, 5
    a, dx, dy = 0.44, 0.22, 0.17
    fig, ax = canvas(18.6, 5.3, 0.7)
    def cube(x0, y0, sel, c):
        def poly(pts, on, shade):
            col = (c[1] if on else '#ffffff')
            ax.add_patch(Polygon(pts, closed=True, fc=col, ec='#868e96' if not on else 'white', lw=0.9, alpha=(shade if on else 1.0)))
        for n in range(NN):
            for h in range(HH):   # front face
                x, y = x0 + n * a, y0 + h * a
                poly([(x, y), (x + a, y), (x + a, y + a), (x, y + a)], sel(n, 0), 0.85)
            for k in range(CC):   # top face
                x, y = x0 + n * a + k * dx, y0 + HH * a + k * dy
                poly([(x, y), (x + a, y), (x + a + dx, y + dy), (x + dx, y + dy)], sel(n, k), 0.6)
        for k in range(CC):       # right face
            for h in range(HH):
                x, y = x0 + NN * a + k * dx, y0 + h * a + k * dy
                poly([(x, y), (x + dx, y + dy), (x + dx, y + dy + a), (x, y + a)], sel(NN - 1, k), 1.0)
        label(ax, x0 + NN * a / 2, y0 - 0.3, 'N（批）', 9.5, '#495057')
        label(ax, x0 - 0.42, y0 + HH * a / 2, 'H, W', 9.5, '#495057')
        label(ax, x0 + NN * a + CC * dx / 2 + 0.5, y0 + CC * dy / 2 - 0.3, 'C（通道）', 9.5, '#495057')
    panels = [
        ('BatchNorm', '每个通道\n在 (N, H, W) 上统计', lambda n, k: k == 0, BLUE),
        ('LayerNorm / RMSNorm', '每个样本\n在 (C, H, W) 上统计', lambda n, k: n == 0, ORANGE),
        ('InstanceNorm', '每个样本的每个通道\n在 (H, W) 上统计', lambda n, k: n == 0 and k == 0, GREEN),
        ('GroupNorm', '每个样本的每组通道\n在 (C/G, H, W) 上统计', lambda n, k: n == 0 and k < 2, PURPLE),
    ]
    for i, (name, desc, sel, c) in enumerate(panels):
        x0 = 0.9 + i * 4.55
        cube(x0, 1.75, sel, c)
        label(ax, x0 + 1.5, 0.95, name, 12, bold=True)
        label(ax, x0 + 1.5, 0.38, desc, 9.5, '#495057')
    save(fig, T, 'norm_dims.png')

def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.grid(color='#e9ecef', lw=0.8); ax.set_axisbelow(True)

# ---- 2. running mean
if 'momentum' in which:
    d = json.load(open(W + 'momentum.json'))
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    it = np.arange(1, 301)
    ax.plot(it, d['batch0.1'], color='#ced4da', lw=0.8, label='每一批的均值')
    ax.axhline(5, color='#868e96', lw=1, ls='--', label='数据的真实均值')
    ax.plot(it, d['0.1'], color=BLUE[1], lw=2, label='running_mean，momentum=0.1（默认）')
    ax.plot(it, d['0.01'], color=ORANGE[1], lw=2, label='running_mean，momentum=0.01')
    ax.set_xlabel('训练模式下前向传播的次数'); ax.set_ylabel('均值'); ax.set_xlim(0, 300); ax.set_ylim(0, 6.6)
    ax.legend(frameon=False, fontsize=9.5, loc='lower right'); style(ax)
    ax.set_title('运行均值从 0 开始，逐步靠近数据的真实均值', fontsize=11.5)
    fig.tight_layout(); save(fig, T, 'running_mean.png')

# ---- 3. batch size
if 'batch' in which:
    d = json.load(open(W + 'batch_size.json'))
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    bss = [2, 4, 8, 16, 32, 64]
    for name, c, mk in [('BatchNorm', BLUE[1], 'o'), ('GroupNorm', PURPLE[1], 's'), ('LayerNorm', ORANGE[1], '^')]:
        m = [np.mean(d[name][str(b)]) * 100 for b in bss]
        ax.plot(range(len(bss)), m, color=c, lw=2, marker=mk, ms=6, label=name)
        for i, b in enumerate(bss):
            ax.scatter([i] * 3, np.array(d[name][str(b)]) * 100, color=c, s=10, alpha=0.35, zorder=1)
    ax.annotate('批大小为 2 时\nBatchNorm 只有 %.1f%%' % (np.mean(d['BatchNorm']['2']) * 100), (0, np.mean(d['BatchNorm']['2']) * 100),
                (0.55, 50), fontsize=9.5, color=BLUE[1], arrowprops=dict(arrowstyle='->', color=BLUE[1]))
    ax.set_xticks(range(len(bss))); ax.set_xticklabels(bss); ax.set_xlabel('训练时的批大小'); ax.set_ylabel('验证准确率（%）')
    ax.set_ylim(30, 100); ax.legend(frameon=False, fontsize=10, loc='lower right'); style(ax)
    ax.set_title('不同批大小下三种归一化层的验证准确率（3 个随机种子的平均）', fontsize=11.5)
    fig.tight_layout(); save(fig, T, 'batch_size.png')

# ---- 5. dropout schematic
if 'dropout' in which:
    fig, ax = canvas(13.2, 4.5, 0.74)
    keep = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    rows = [('输入', ['1.00'] * 10, [1] * 10, GREY), ('训练模式\ntrain()', ['1.43' if k else '0' for k in keep], keep, ORANGE), ('推理模式\neval()', ['1.00'] * 10, [1] * 10, GREEN)]
    for r, (name, vals, on, c) in enumerate(rows):
        y = 3.35 - r * 1.25
        label(ax, 1.25, y + 0.35, name, 10.5, bold=True)
        for i, (v, k) in enumerate(zip(vals, on)):
            box(ax, 2.6 + i * 0.95, y, 0.8, 0.7, v, c if k else ('#ffffff', '#ced4da'), fs=10, tc=INK if k else '#adb5bd', ls='-' if k else '--')
    label(ax, 12.25, 2.45, '均值 1.00', 9.5, '#868e96', ha='left'); label(ax, 12.25, 3.7, '均值 1.00', 9.5, '#868e96', ha='left'); label(ax, 12.25, 1.2, '均值 1.00', 9.5, '#868e96', ha='left')
    label(ax, 7.3, 0.4, 'p = 0.3：训练时随机把 30% 的值置零，其余的除以 1 − p = 0.7；推理时原样通过', 10, '#495057')
    ax.set_xlim(0, 13.6); fig.set_size_inches(13.6 * 0.74, 4.5 * 0.74)
    save(fig, T, 'dropout.png')

# ---- 4. overfitting curves
if 'overfit' in which:
    h = np.array(json.load(open(W + 'overfit.json'))); ep = np.arange(1, len(h) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axs[0]
    ax.plot(ep, h[:, 0], color=BLUE[1], lw=2, label='训练损失'); ax.plot(ep, h[:, 1], color=RED[1], lw=2, label='验证损失')
    b = h[:, 1].argmin(); ax.axvline(b + 1, color='#adb5bd', lw=1, ls='--')
    ax.annotate('验证损失最低\n第 %d 轮，%.3f' % (b + 1, h[b, 1]), (b + 1, h[b, 1]), (b + 28, 1.0), fontsize=9.5, color='#495057', arrowprops=dict(arrowstyle='->', color='#868e96'))
    ax.set_xlabel('轮数'); ax.set_ylabel('损失'); ax.set_ylim(0, 2.3); ax.set_xlim(0, 200); ax.legend(frameon=False, fontsize=10); style(ax)
    ax.set_title('损失：训练损失趋于 0，验证损失先降后升', fontsize=11)
    ax = axs[1]
    ax.plot(ep, h[:, 2] * 100, color=BLUE[1], lw=2, label='训练准确率'); ax.plot(ep, h[:, 3] * 100, color=RED[1], lw=2, label='验证准确率')
    ax.set_xlabel('轮数'); ax.set_ylabel('准确率（%）'); ax.set_ylim(30, 102); ax.set_xlim(0, 200); ax.legend(frameon=False, fontsize=10, loc='lower right'); style(ax)
    ax.set_title('准确率：训练集 100%%，验证集停在 %.1f%%' % (h[-1, 3] * 100), fontsize=11)
    fig.tight_layout(w_pad=2.5); save(fig, T, 'overfitting.png')

# ---- 6. comparison
if 'compare' in which:
    d = json.load(open(W + 'compare.json'))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0))
    cols = {'不使用正则化': '#868e96', '权重衰减': BLUE[1], 'Dropout': ORANGE[1], '两者同时使用': PURPLE[1]}
    labels = {'不使用正则化': '不使用正则化', '权重衰减': '权重衰减 1.0', 'Dropout': 'Dropout 0.5', '两者同时使用': '两者同时使用'}
    for name, h in d.items():
        h = np.array(h); ep = np.arange(1, len(h) + 1)
        axs[0].plot(ep, h[:, 1], color=cols[name], lw=2, label=labels[name])
        axs[1].plot(ep, h[:, 3] * 100, color=cols[name], lw=1.8, label=labels[name])
    axs[0].set_ylim(0.3, 0.8); axs[0].set_ylabel('验证损失'); axs[0].set_title('验证损失', fontsize=11)
    axs[1].set_ylim(80, 92); axs[1].set_ylabel('验证准确率（%）'); axs[1].set_title('验证准确率', fontsize=11)
    for ax in axs: ax.set_xlim(0, 200); ax.set_xlabel('轮数'); style(ax)
    axs[0].legend(frameon=False, fontsize=10, loc='upper right')
    fig.tight_layout(w_pad=2.5); save(fig, T, 'comparison.png')
