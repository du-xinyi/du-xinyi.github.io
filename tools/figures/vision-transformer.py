import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json
from lib import *
import numpy as np
T = '2026-11-12-vision-transformer'
W = HERE + '/data/vision-transformer/'
which = sys.argv[1:] or ['patches', 'data_size', 'attention', 'cost']
att = np.load(W + 'attention.npz')

# ---- 1. patches：从图像到词元序列，再到分类结果
if 'patches' in which:
    fig, ax = canvas(16.0, 7.3, 0.78)
    img = att['images'][0]
    # 原图与切块的网格
    label(ax, 1.7, 6.95, '8x8 的图像，切成 4x4 个 2x2 的图像块', 9.6)
    ax.imshow(img, cmap='Greys', extent=(0.3, 3.1, 3.6, 6.4), vmin=0, vmax=1, zorder=1)
    for k in range(5):
        ax.plot([0.3 + 0.7 * k] * 2, [3.6, 6.4], color=ORANGE[1], lw=1.3, zorder=2)
        ax.plot([0.3, 3.1], [3.6 + 0.7 * k] * 2, color=ORANGE[1], lw=1.3, zorder=2)
    # 展开成序列的图像块
    x0, y0, s, gap = 4.6, 4.75, 0.5, 0.14
    label(ax, x0 + 8 * (s + gap) - gap / 2, 6.95, '按从左到右、从上到下的顺序排成序列，每个图像块展平为 C·P·P = 4 个数', 9.6)
    for k in range(16):
        r, c = divmod(k, 4)
        x = x0 + k * (s + gap)
        ax.imshow(img[2 * r:2 * r + 2, 2 * c:2 * c + 2], cmap='Greys', extent=(x, x + s, y0 + 1.0, y0 + 1.0 + s), vmin=0, vmax=1, zorder=1)
        ax.add_patch(Rectangle((x, y0 + 1.0), s, s, fc='none', ec=ORANGE[1], lw=1.2, zorder=2))
    arrow(ax, (3.2, 5.0), (4.45, 6.0), lw=1.3)
    # 线性变换
    xe = x0 + 16 * (s + gap) - gap
    box(ax, x0, y0 + 0.2, xe - x0, 0.5, '线性变换（卷积核大小和步幅都为 P 的卷积）：4 维 → E 维', GREEN, fs=9.6)
    # 词元
    yt = y0 - 0.95
    xc = x0 - (s + gap)
    box(ax, xc, yt, s, 0.75, 'CLS', PURPLE, fs=7.6, round=0.05)
    for k in range(16):
        x = x0 + k * (s + gap)
        box(ax, x, yt, s, 0.75, str(k + 1), BLUE, fs=8.4, round=0.05)
        arrow(ax, (x + s / 2, y0 + 0.2), (x + s / 2, yt + 0.75), lw=0.9)
    label(ax, xc - 0.15, yt + 0.37, '分类标记\n可学习', 8.6, PURPLE[1], ha='right')
    # 位置嵌入
    yp = yt - 0.95
    for k in range(17):
        x = xc + k * (s + gap)
        box(ax, x, yp, s, 0.5, str(k), ORANGE, fs=8.0, round=0.05)
        label(ax, x + s / 2, yp + 0.72, '+', 10.5)
    label(ax, xc - 0.15, yp + 0.25, '位置嵌入\n可学习', 8.6, ORANGE[1], ha='right')
    # 编码器
    ye = yp - 1.25
    box(ax, xc, ye, xe - xc, 0.75, 'Transformer 编码器层 × L        输入与输出都是 (N, 17, E)', BLUE, fs=10)
    for k in range(17):
        x = xc + k * (s + gap) + s / 2
        arrow(ax, (x, yp), (x, ye + 0.75), lw=0.9)
    # 输出
    yo = ye - 1.05
    box(ax, xc, yo, s, 0.6, 'CLS', PURPLE, fs=7.6, round=0.05)
    arrow(ax, (xc + s / 2, ye), (xc + s / 2, yo + 0.6), lw=1.1)
    for k in range(16):
        x = x0 + k * (s + gap)
        box(ax, x, yo, s, 0.6, '', GREY, round=0.05, ls='--', lw=1.0)
        arrow(ax, (x + s / 2, ye), (x + s / 2, yo + 0.6), lw=0.7, color='#adb5bd')
    label(ax, x0 + 8 * (s + gap), yo - 0.33, '图像块位置的输出在分类时不使用', 8.8, '#868e96')
    box(ax, 0.3, yo, 2.3, 0.6, 'LayerNorm + 线性层', GREEN, fs=9.2)
    arrow(ax, (xc, yo + 0.3), (2.6, yo + 0.3), lw=1.3)
    box(ax, 0.3, yo + 1.2, 2.3, 0.6, '10 个类别的 logits', BLUE, fs=9.2)
    arrow(ax, (1.45, yo + 0.6), (1.45, yo + 1.2), lw=1.3)
    ax.set_ylim(-0.25, 7.3)
    save(fig, T, 'patches.png')

# ---- 2. data_size：准确率随训练集大小的变化
if 'data_size' in which:
    res = json.load(open(W + 'data_size.json'))
    sizes = res['sizes']
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    sty = {'CNN': ('CNN（56,714 个参数）', '#2e9950', 's'), 'ViT': ('ViT（69,194 个参数）', '#2f6fd0', 'o'),
           'ViT 无位置嵌入': ('ViT，去掉位置嵌入', '#868e96', '^')}
    for k, (name, c, m) in sty.items():
        a = np.array(res['accuracy'][k])
        ax.fill_between(sizes, a.min(1), a.max(1), color=c, alpha=0.15, lw=0)
        ax.plot(sizes, a.mean(1), color=c, marker=m, ms=5, lw=1.8, label=name)
        for x, y in zip(sizes, a.mean(1)):
            ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 7 if k != 'ViT 无位置嵌入' else -14),
                        ha='center', fontsize=8, color=c)
    ax.set_xscale('log'); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes]); ax.minorticks_off()
    ax.set_xlabel('训练集的图像数量'); ax.set_ylabel('测试集准确率'); ax.set_ylim(0.3, 1.06); ax.grid(alpha=0.3)
    ax.set_title('8x8 手写数字上的测试集准确率（3 个随机种子的平均值，色带为最小值到最大值）', fontsize=11)
    ax.legend(loc='lower right', frameon=False)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout(); save(fig, T, 'data_size.png')

# ---- 3. attention：分类标记在最后一层关注的图像块
if 'attention' in which:
    fig, axs = plt.subplots(2, 10, figsize=(12.4, 3.25))
    for k in range(10):
        ok = att['labels'][k] == att['predictions'][k]
        axs[0, k].imshow(att['images'][k], cmap='Greys', vmin=0, vmax=1)
        axs[0, k].set_title(f'预测 {att["predictions"][k]}', fontsize=10, color=INK if ok else RED[1])
        im = axs[1, k].imshow(att['maps'][k], cmap='Blues', vmin=0, vmax=0.5, extent=(-0.5, 7.5, 7.5, -0.5))
        ys, xs = np.nonzero(att['images'][k] > 0.5)
        axs[1, k].scatter(xs, ys, s=5, color=ORANGE[1], lw=0)
        for a in axs[:, k]:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_color('#adb5bd')
    axs[0, 0].set_ylabel('输入', fontsize=10); axs[1, 0].set_ylabel('注意力', fontsize=10)
    fig.subplots_adjust(left=0.035, right=0.93, top=0.9, bottom=0.03, wspace=0.08, hspace=0.08)
    cax = fig.add_axes([0.94, 0.05, 0.008, 0.4]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8)
    save(fig, T, 'attention.png')

# ---- 4. cost：分辨率与耗时、显存
if 'cost' in which:
    res = json.load(open(W + 'cost.json'))
    L = np.array([r['tokens'] for r in res]); names = [f"{r['image_size']}\n{r['tokens']}" for r in res]
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axs[0]
    ms = np.array([r['model_ms'] for r in res])
    ax.plot(L, ms[0] * L / L[0], color='#adb5bd', ls='--', lw=1.2, label='与词元数成正比')
    ax.plot(L, ms[0] * (L / L[0]) ** 2, color='#adb5bd', ls=':', lw=1.4, label='与词元数的平方成正比')
    ax.plot(L, ms, color='#2f6fd0', marker='o', ms=5, lw=1.8, label='ViT-B/16 前向传播')
    for x, y in zip(L, ms): ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8.5, color='#2f6fd0')
    fast = np.array([r['fast_ms'] for r in res])
    ax.plot(L, fast[0] * L / L[0], color='#adb5bd', ls='--', lw=1.2)
    ax.plot(L, fast[0] * (L / L[0]) ** 2, color='#adb5bd', ls=':', lw=1.4)
    ax.plot(L, fast, color='#2e9950', marker='s', ms=5, lw=1.8, label='一层注意力，不返回权重')
    for x, y in zip(L, fast): ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8.5, color='#2e9950')
    ax.set_ylabel('每批 2 张图像的耗时（ms）'); ax.set_title('耗时', fontsize=11.5); ax.set_ylim(0.05, 3000)
    ax = axs[1]
    for key, name, c, m in [('full_mb', '一层注意力，返回每个头的权重', '#d64545', 'x'), ('model_mb', 'ViT-B/16 前向传播', '#2f6fd0', 'o'),
                            ('fast_mb', '一层注意力，不返回权重', '#2e9950', 's')]:
        v = np.array([r[key] for r in res])
        ax.plot(L, v, color=c, marker=m, ms=5, lw=1.8, label=name)
        ax.annotate(f'{v[-1]:.0f}', (L[-1], v[-1]), textcoords='offset points', xytext=(-6, 7), ha='right', fontsize=8.5, color=c)
    ax.set_ylabel('额外占用的显存峰值（MB）'); ax.set_title('显存', fontsize=11.5)
    for ax in axs:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(L); ax.set_xticklabels(names, fontsize=9); ax.minorticks_off()
        ax.set_xlabel('输入边长（上）与词元数（下）'); ax.grid(alpha=0.3); ax.legend(loc='upper left', frameon=False, fontsize=9)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout(); save(fig, T, 'cost.png')
