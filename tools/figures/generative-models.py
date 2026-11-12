import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys
from lib import *
import numpy as np
T = '2026-11-11-generative-models'
W = HERE + '/data/generative-models/'
which = sys.argv[1:] or ['overview', 'vae', 'gan', 'forward', 'diffusion', 'cfg']


def strip(ax, imgs, vmin=0, vmax=1, pad=1):
    """把若干张 8x8 图像横向拼成一条，中间留白"""
    imgs = np.asarray(imgs).reshape(-1, 8, 8)
    n = len(imgs)
    out = np.full((8, n * 8 + (n - 1) * pad), np.nan)
    for i, im in enumerate(imgs):
        out[:, i * (8 + pad): i * (8 + pad) + 8] = im
    cmap = plt.get_cmap('gray_r').copy(); cmap.set_bad('white')
    ax.imshow(out, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def grid_img(ax, imgs, rows, cols, vmin=0, vmax=1, pad=1):
    imgs = np.asarray(imgs).reshape(rows, cols, 8, 8)
    out = np.full((rows * 8 + (rows - 1) * pad, cols * 8 + (cols - 1) * pad), np.nan)
    for r in range(rows):
        for c in range(cols):
            out[r * (8 + pad): r * (8 + pad) + 8, c * (8 + pad): c * (8 + pad) + 8] = imgs[r, c]
    cmap = plt.get_cmap('gray_r').copy(); cmap.set_bad('white')
    ax.imshow(out, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


# ---- 1. overview
if 'overview' in which:
    fig, ax = canvas(15.2, 9.3, 0.8)
    H = 0.8

    def row(y, title, sub):
        label(ax, 0.25, y + H / 2 + 0.2, title, 11.5, bold=True, ha='left')
        label(ax, 0.25, y + H / 2 - 0.28, sub, 8.6, '#868e96', ha='left')

    def chain(y, items, x0=3.3, gap=0.55):
        x = x0; prev = None
        for text, c, w in items:
            box(ax, x, y, w, H, text, c, fs=9.4)
            if prev is not None: arrow(ax, (prev, y + H / 2), (x, y + H / 2), lw=1.2)
            prev = x + w; x += w + gap
        return x

    # VAE
    y = 7.9
    row(y, 'VAE', '编码成分布，再解码')
    chain(y, [('真实样本 $x$', GREY, 1.6), ('编码器', BLUE, 1.3), ('均值、方差', ORANGE, 1.6), ('采样 $z$', PURPLE, 1.3), ('解码器', GREEN, 1.3), ('重建 $\\hat{x}$', GREY, 1.4)], gap=0.45)
    label(ax, 10.4, y - 0.38, '生成时：直接从标准正态分布采样 $z$，只用解码器', 8.8, '#7a4fd0')
    ax.plot([0.2, 15.0], [y - 0.75, y - 0.75], color='#dee2e6', lw=1)

    # GAN
    y = 5.55
    row(y, 'GAN', '生成器与判别器对抗')
    chain(y, [('随机噪声 $z$', PURPLE, 1.7), ('生成器', GREEN, 1.5), ('生成的样本', GREY, 1.7)])
    box(ax, 11.0, y - 0.5, 1.5, H, '判别器', BLUE, fs=9.4); box(ax, 13.05, y - 0.5, 1.6, H, '真 / 假', ORANGE, fs=9.4)
    box(ax, 7.4, y - 1.05, 1.7, H, '真实样本', GREY, fs=9.4)
    arrow(ax, (9.1, y + H / 2), (11.0, y + 0.05), lw=1.2); arrow(ax, (9.1, y - 1.05 + H / 2), (11.0, y - 0.25), lw=1.2)
    arrow(ax, (12.5, y - 0.1), (13.05, y - 0.1), lw=1.2)
    label(ax, 5.0, y - 0.62, '生成时：只用生成器，一次前向传播', 8.8, '#7a4fd0')
    ax.plot([0.2, 15.0], [y - 1.4, y - 1.4], color='#dee2e6', lw=1)

    # diffusion
    y = 2.85
    row(y, '扩散模型', '逐步加噪，学习逐步去噪')
    xs = [3.3, 5.75, 8.2, 10.65, 13.1]; names = ['$x_0$\n真实样本', '$x_1$', '…', '$x_{T-1}$', '$x_T$\n纯噪声']
    for i, (x, n) in enumerate(zip(xs, names)):
        box(ax, x, y, 1.5, H, n, [GREY, GREY, GREY, GREY, PURPLE][i], fs=9.4)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 1.5, y + H * 0.72), (b, y + H * 0.72), lw=1.2, color='#868e96')
        arrow(ax, (b, y + H * 0.28), (a + 1.5, y + H * 0.28), lw=1.2, color='#2e9950')
    label(ax, 9.0, y + H + 0.3, '前向过程：每一步加一点高斯噪声，不需要训练', 8.8, '#868e96')
    label(ax, 9.0, y - 0.38, '生成时：从纯噪声出发，用网络预测噪声并一步步去掉，需要多次前向传播', 8.8, '#2e9950')
    ax.plot([0.2, 15.0], [y - 0.8, y - 0.8], color='#dee2e6', lw=1)

    # autoregressive
    y = 0.75
    row(y, '自回归模型', '逐个预测下一个元素')
    chain(y, [('已生成的部分\n$x_1, \\ldots, x_{i-1}$', GREY, 2.5), ('模型', BLUE, 1.5), ('下一个元素的\n概率分布', ORANGE, 2.2), ('采样 $x_i$', PURPLE, 1.6)])
    ax.plot([12.2, 12.2, 4.55, 4.55], [y, y - 0.4, y - 0.4, y], color='#7a4fd0', lw=1.2)
    arrow(ax, (4.55, y - 0.4), (4.55, y), color='#7a4fd0', lw=1.2)
    label(ax, 8.4, y - 0.62, '接到序列末尾，重复这个过程', 8.8, '#7a4fd0')
    ax.set_ylim(-0.2, 9.3)
    save(fig, T, 'overview.png')

# ---- 2. vae: reconstructions and samples
if 'vae' in which:
    d = np.load(W + 'vae.npz')
    rows = [('真实图像', d['real'], INK), ('自编码器的重建', d['ae_rec'], INK), ('VAE 的重建', d['vae_rec'], INK),
            ('自编码器：解码随机隐变量', d['ae_s'], '#d64545'), ('VAE：解码随机隐变量', d['vae_s'], '#2e9950')]
    fig, axs = plt.subplots(len(rows), 1, figsize=(10.5, 5.6))
    for ax, (name, imgs, col) in zip(axs, rows):
        strip(ax, imgs[:14])
        ax.set_title(name, fontsize=10.5, color=col, loc='left', pad=4)
    fig.tight_layout(h_pad=0.9); save(fig, T, 'vae.png')

# ---- 3. gan
if 'gan' in which:
    d = np.load(W + 'gan.npz')
    steps = [500, 1000, 1500, 3000]
    fig, axs = plt.subplots(2, 5, figsize=(12.5, 5.5))
    for r, (key, name, col) in enumerate([('a', 'betas=(0.5, 0.999)', '#2f6fd0'), ('b', 'betas=(0.9, 0.999)', '#d9822b')]):
        for c in range(5):
            ax = axs[r, c]
            if c == 0:
                ax.scatter(*d['real'].T, s=2, color='#868e96', alpha=0.5, lw=0); ax.set_title('真实样本', fontsize=10.5)
            else:
                ax.scatter(*d['real'].T, s=2, color='#ced4da', alpha=0.35, lw=0)
                ax.scatter(*d[f'{key}{steps[c - 1]}'].T, s=2.5, color=col, alpha=0.6, lw=0)
                ax.set_title(f'第 {steps[c - 1]} 步', fontsize=10.5)
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color('#dee2e6')
        axs[r, 0].set_ylabel(name, fontsize=10.5, color=col, labelpad=8)
    fig.tight_layout(w_pad=0.6, h_pad=1.0); save(fig, T, 'gan.png')

# ---- 4. forward process
if 'forward' in which:
    d = np.load(W + 'diffusion.npz')
    ts = [0, 100, 200, 300, 500, 700, 1000]
    fig = plt.figure(figsize=(10.5, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    f = d['forward'][:, :6]                      # (7 个时刻, 6 张图, 64)
    grid_img(ax, f.transpose(1, 0, 2), 6, 7, vmin=-1, vmax=1)   # 每行一张图，每列一个时刻
    ax.set_xticks([4 + 9 * i - 0.5 for i in range(7)]); ax.set_xticklabels([f'$t={t}$' for t in ts], fontsize=9.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_title('同一批图像在不同步数下的加噪结果', fontsize=11, y=-0.09)
    ax = fig.add_subplot(gs[1])
    ab = d['alpha_bars']; t = np.arange(1, len(ab) + 1)
    ax.plot(t, np.sqrt(ab), color='#2f6fd0', lw=2, label='$x_0$ 的系数 $\\sqrt{\\bar{\\alpha}_t}$')
    ax.plot(t, np.sqrt(1 - ab), color='#d9822b', lw=2, label='噪声的系数 $\\sqrt{1-\\bar{\\alpha}_t}$')
    ax.set_xlabel('步数 $t$'); ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05); ax.grid(color='#e9ecef'); ax.legend(fontsize=9.5, loc='center right')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.set_title('两个系数随步数的变化', fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12); save(fig, T, 'forward.png')

# ---- 5. diffusion sampling
if 'diffusion' in which:
    d = np.load(W + 'diffusion.npz')
    keep = [1000, 600, 400, 200, 100, 50, 0]
    fig = plt.figure(figsize=(10.5, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 4.6], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    tr = d['traj'][:, :6]
    grid_img(ax, np.clip(tr.transpose(1, 0, 2), -1, 1), 6, 7, vmin=-1, vmax=1)
    ax.set_xticks([4 + 9 * i - 0.5 for i in range(7)]); ax.set_xticklabels(['$x_{999}$', '$x_{600}$', '$x_{400}$', '$x_{200}$', '$x_{100}$', '$x_{50}$', '$x_0$'], fontsize=10)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_title('采样过程：每一行是一个样本从噪声逐步变成图像', fontsize=11, y=-0.09)
    ax = fig.add_subplot(gs[1])
    grid_img(ax, np.clip(d['samples'], -1, 1), 4, 4, vmin=-1, vmax=1)
    ax.set_title('生成的 16 个样本', fontsize=11, y=-0.09)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08); save(fig, T, 'diffusion.png')

# ---- 6. classifier-free guidance
if 'cfg' in which:
    d = np.load(W + 'diffusion.npz')
    scales = [0.0, 1.0, 2.0, 5.0]
    names = ['引导强度 0：不使用标签', '引导强度 1：普通的条件生成', '引导强度 2', '引导强度 5：过强，图像被破坏']
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 5.7))
    for ax, s, n in zip(axs.flat, scales, names):
        grid_img(ax, np.clip(d[f'cfg{s}'][:40], -1, 1), 4, 10, vmin=-1, vmax=1)
        ax.set_xticks([4 + 9 * i - 0.5 for i in range(10)]); ax.set_xticklabels([str(i) for i in range(10)], fontsize=10)
        ax.xaxis.tick_top(); ax.tick_params(length=0)
        ax.set_title(n, fontsize=11, y=-0.14)
    fig.tight_layout(w_pad=2.5, h_pad=1.5); save(fig, T, 'cfg.png')
