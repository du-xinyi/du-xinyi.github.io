import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json
from lib import *
import numpy as np
from matplotlib.lines import Line2D
T = '2026-11-12-keypoint-detection'
W = HERE + '/data/keypoint-detection/'
D = np.load(W + 'fig.npz')
J = json.load(open(W + 'fig.json'))
NAMES = ['头', '左翼', '右翼', '尾']
KC = [RED[1], BLUE[1], ORANGE[1], GREEN[1]]

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#adb5bd')

def show(ax, img):
    ax.imshow(img, interpolation='nearest'); ax.set_xlim(-0.5, 63.5); ax.set_ylim(63.5, -0.5); clean(ax)

def mark(ax, kpts, vis=None, names=NAMES, text=True, ms=7):
    for k, (x, y) in enumerate(kpts):
        v = 2 if vis is None else vis[k]
        if v == 0: continue
        ax.plot(x, y, 'o', ms=ms, mfc=KC[k] if v == 2 else 'none', mec='white' if v == 2 else KC[k], mew=1.0 if v == 2 else 2.0, clip_on=False)
        if text:
            dx = 4 if x < 50 else -4
            ax.text(x + dx, y - 3 if y > 8 else y + 5, names[k], color='white', fontsize=9, fontweight='bold', ha='left' if dx > 0 else 'right', va='center',
                    bbox=dict(fc=KC[k], ec='none', pad=1.2, alpha=0.9))

# ---- 1. samples and heatmap targets
fig, axs = plt.subplots(2, 4, figsize=(11.5, 6.1))
subt = ['4 个关键点都可见', '4 个关键点都可见', '空心圆：被遮挡（v = 1）', '图像之外的关键点（v = 0）不画']
for i, ax in enumerate(axs[0]):
    show(ax, D['s_img'][i]); mark(ax, D['s_kpts'][i], D['s_vis'][i]); ax.set_title(subt[i], fontsize=10.5)
for k, ax in enumerate(axs[1]):
    ax.imshow(D['s_hm'][k], cmap='magma', vmin=0, vmax=1, interpolation='nearest'); clean(ax)
    u = (D['s_kpts'][0][k] + 0.5) / 4 - 0.5
    ax.plot(u[0], u[1], '+', color='white', ms=11, mew=1.3)
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True); ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which='minor', color='#495057', lw=0.3); ax.tick_params(which='minor', length=0)
    ax.set_title(f'第 1 张图像“{NAMES[k]}”的目标热力图 16x16', fontsize=10.5, color=KC[k])
fig.tight_layout(w_pad=1.0, h_pad=1.2); save(fig, T, 'samples.png')

# ---- 2. decoding
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
true_u = 6.42; cells = np.arange(3, 11); vals = np.exp(-(cells - true_u) ** 2 / 2)
xs = np.linspace(2.5, 10.5, 300)
ax.plot(xs, np.exp(-(xs - true_u) ** 2 / 2), color=GREY[1], lw=1.2, ls='--', label='连续的高斯函数')
ax.bar(cells, vals, width=0.9, color=BLUE[0], ec=BLUE[1], lw=1.2, label='热力图上的值')
peak = cells[vals.argmax()]
l, c, r = np.log(vals[vals.argmax() - 1]), np.log(vals.max()), np.log(vals[vals.argmax() + 1])
para = peak + 0.5 * (l - r) / (l + r - 2 * c)
for x, col, name, y, right in [(peak, RED[1], 'argmax', 1.12, False), (peak + 0.25, ORANGE[1], '1/4 偏移', 1.27, False), (para, GREEN[1], '真实位置 = 抛物线拟合', 1.2, True)]:
    ax.plot([x, x], [0, y - 0.05], color=col, lw=1.6)
    ax.text(x + (0.1 if right else -0.1), y, name, color=col, fontsize=9.5, ha='left' if right else 'right', va='center')
ax.set_ylim(0, 1.38); ax.set_xlim(2.4, 10.6); ax.set_xticks(cells); ax.set_xlabel('热力图的列（1 格 = 输入图像的 4 个像素）'); ax.set_ylabel('热力图的值')
ax.set_title('一行热力图上的解码（sigma = 1）', fontsize=11); ax.legend(loc='upper right', fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.78))
ms = [('argmax', 'argmax'), ('quarter', '1/4 偏移'), ('parabola', '抛物线拟合'), ('soft', 'soft-argmax')]
cfg = [('4-1.0', 'stride 4, sigma 1', BLUE[1]), ('4-2.0', 'stride 4, sigma 2', ORANGE[1]), ('2-2.0', 'stride 2, sigma 2', GREEN[1])]
for ax, j, t in [(axs[1], 0, '无噪声的热力图'), (axs[2], 1, '加上标准差 0.05 的噪声')]:
    for ci, (ck, cn, col) in enumerate(cfg):
        v = [J['decode'][f'{ck}-{m}'][j] for m, _ in ms]
        b = ax.bar(np.arange(4) + (ci - 1) * 0.27, v, width=0.25, color=col, label=cn)
        for x, y in zip(np.arange(4) + (ci - 1) * 0.27, v): ax.text(x, y + 0.03, f'{y:.2f}', ha='center', va='bottom', fontsize=7, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in ms], fontsize=9.5); ax.set_ylim(0, 2.35); ax.set_title(t, fontsize=11)
    ax.grid(axis='y', color='#e9ecef'); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axs[1].set_ylabel('平均解码误差（输入图像的像素）'); axs[1].legend(fontsize=8.5, frameon=False, loc='upper right')
fig.tight_layout(w_pad=1.6); save(fig, T, 'decoding.png')

# ---- 3. predictions
sel = [0, 1, 2, 3, 4]
fig, axs = plt.subplots(1, 5, figsize=(13, 3.15))
for ax, i in zip(axs, sel):
    show(ax, D['p_img'][i])
    vis = D['p_vis'][i]; gt = D['p_kpts'][i]; lab = vis > 0
    ax.plot(gt[lab, 0], gt[lab, 1], 'o', ms=9, mfc='none', mec='white', mew=1.6)
    ax.plot(D['p_h'][i][lab, 0], D['p_h'][i][lab, 1], 'x', ms=7, color='#40c057', mew=2)
    ax.plot(D['p_r'][i][lab, 0], D['p_r'][i][lab, 1], '+', ms=9, color='#ff922b', mew=2)
    eh = np.linalg.norm(D['p_h'][i] - gt, axis=1)[lab].mean(); er = np.linalg.norm(D['p_r'][i] - gt, axis=1)[lab].mean()
    ax.set_title(f'热力图 {eh:.2f}  回归 {er:.2f}', fontsize=10.5)
fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc='none', mec=INK, mew=1.6, ms=9), Line2D([], [], marker='x', ls='', color='#40c057', mew=2, ms=7), Line2D([], [], marker='+', ls='', color='#ff922b', mew=2, ms=9)],
           labels=['标注', '热力图模型（抛物线拟合解码）', '坐标回归模型'], loc='lower center', ncol=3, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0); save(fig, T, 'predictions.png')

# ---- 4. PCK curve
fig, ax = plt.subplots(figsize=(7.2, 3.9))
a = np.array(J['alphas'])
for k, n, col, ls in [('parabola', '热力图 + 抛物线拟合', GREEN[1], '-'), ('argmax', '热力图 + argmax', BLUE[1], '--'), ('regression', '坐标回归', ORANGE[1], '-')]:
    ax.plot(a, J[k], color=col, lw=2, ls=ls, label=n)
for x in [0.05, 0.1]:
    ax.axvline(x, color=GREY[1], lw=1, ls=':'); ax.text(x + 0.002, 0.03, f'PCK@{x}', fontsize=9, color=GREY[1])
ax.set_xlabel('阈值（物体长度的倍数）'); ax.set_ylabel('PCK'); ax.set_xlim(0, 0.2); ax.set_ylim(0, 1.02)
ax.grid(color='#e9ecef'); ax.legend(frameon=False, fontsize=10, loc='lower right')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.tight_layout(); save(fig, T, 'pck.png')

# ---- 5. flip
img, kp = D['f_img'], D['f_kpts']
wrong = kp.copy(); wrong[:, 0] = 63 - wrong[:, 0]; right = wrong[[0, 2, 1, 3]]
fig, axs = plt.subplots(1, 5, figsize=(13.5, 3.45))
for ax, im, k, t, tc in [(axs[0], img, kp, '原图与标注\n', INK), (axs[1], img[:, ::-1], wrong, '错误：只翻转 x 坐标\n“左翼”落在了右侧', RED[1]), (axs[2], img[:, ::-1], right, '正确：再互换左翼和右翼\n', GREEN[1])]:
    show(ax, im); mark(ax, k); ax.set_title(t, fontsize=10.5, color=tc)
for ax, h, t, tc in [(axs[3], D['f_hm_wrong'][1], '用错误的翻转训练\n“左翼”通道有两个峰', RED[1]), (axs[4], D['f_hm_right'][1], '不做翻转训练\n“左翼”通道只有一个峰', GREEN[1])]:
    ax.imshow(np.clip(h, 0, 1), cmap='magma', vmin=0, vmax=0.8, interpolation='nearest'); clean(ax); ax.set_title(t, fontsize=10.5, color=tc)
    u = (kp[1] + 0.5) / 4 - 0.5
    ax.plot(u[0], u[1], 'o', ms=13, mfc='none', mec='white', mew=1.2)
fig.tight_layout(w_pad=1.0); save(fig, T, 'flip.png')
