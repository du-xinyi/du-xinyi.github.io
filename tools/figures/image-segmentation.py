import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.image as mpimg
T = '2026-11-12-image-segmentation'
W = HERE + '/data/image-segmentation/'
D = np.load(W + 'fig.npz')
CLS = ['背景', '圆', '矩形', '三角形']
CC = ['#343a40', RED[1], BLUE[1], GREEN[1]]
cmap = ListedColormap(CC); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#adb5bd')

# ---- 1. task types
img, sem, inst = D['t_img'], D['t_sem'], D['t_inst']
semc = np.array([[52, 58, 64], [214, 69, 69], [47, 111, 208], [122, 140, 90]])[sem]
ic = np.array([[52, 58, 64], [217, 130, 43], [122, 79, 208], [47, 111, 208]])
insc = ic[inst].copy(); insc[inst == 0] = [233, 236, 239]
pan = ic[inst].copy(); pan[(inst == 0) & (sem == 3)] = [122, 140, 90]; pan[(inst == 0) & (sem == 0)] = [52, 58, 64]
fig, axs = plt.subplots(1, 4, figsize=(13, 2.85))
titles = ['输入图像', '语义分割\n每个像素一个类别，两个圆同色', '实例分割\n每个目标一个掩码，只管可数的目标', '全景分割\n可数的目标分实例，地面等区域只分类别']
for ax, im, t in zip(axs, [img, semc, insc, pan], titles):
    ax.imshow(im.astype(np.uint8)); clean(ax); ax.set_title(t, fontsize=10.5)
for ax in axs[2:]:
    for k, n in [(1, '圆 #1'), (2, '圆 #2'), (3, '矩形 #1')]:
        ys, xs = np.nonzero(inst == k); ax.text(xs.mean(), ys.mean(), n, ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
axs[3].text(120, 140, '地面', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
axs[1].text(120, 140, '地面', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
fig.tight_layout(w_pad=1.2); save(fig, T, 'task_types.png')

# ---- 2. interpolation
fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.2))
cmap5 = ListedColormap(['#e9ecef', ORANGE[1], PURPLE[1], GREEN[1]])
for ax, m, t, z in [(axs[0], D['i_mask'], '原始掩码 64x64（局部）', (10, 22)), (axs[1], D['i_bil'], '双线性插值到 100x100 后取整（局部）', (16, 34)), (axs[2], D['i_near'], '最近邻插值到 100x100（局部）', (16, 34))]:
    a, b = z; sub = m[a:b, a:b]
    ax.imshow(sub, cmap=cmap5, norm=norm); clean(ax); ax.set_title(t, fontsize=10.5)
    for (y, x), v in np.ndenumerate(sub):
        ax.text(x, y, str(v), ha='center', va='center', fontsize=6.5 if b - a > 12 else 8, color='white' if v else '#868e96')
fig.legend(handles=[Patch(fc=c, ec='none') for c in cmap5.colors], labels=['0 背景', '1 圆（凭空出现）', '2 矩形（凭空出现）', '3 三角形'],
           loc='lower center', ncol=4, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=1.5); save(fig, T, 'interpolation.png')

# ---- 3. U-Net
fig, ax = canvas(13, 6.2)
enc = [('enc1', '16 x 64 x 64', 0.5, 4.9), ('enc2', '32 x 32 x 32', 2.1, 3.6), ('enc3', '64 x 16 x 16', 3.7, 2.3)]
dec = [('dec1', '16 x 64 x 64', 10.5, 4.9), ('dec2', '32 x 32 x 32', 8.9, 3.6), ('dec3', '64 x 16 x 16', 7.3, 2.3)]
label(ax, 1.4, 6.05, '输入 3 x 64 x 64', fs=10)
for n, s, x, y in enc: box(ax, x, y, 1.8, 0.8, f'{n}\n{s}', BLUE, fs=9.5)
for n, s, x, y in dec: box(ax, x, y, 1.8, 0.8, f'{n}\n{s}', GREEN, fs=9.5)
box(ax, 5.5, 1.0, 2.0, 0.8, 'bottleneck\n128 x 8 x 8', PURPLE, fs=9.5)
arrow(ax, (1.4, 5.93), (1.4, 5.72))
for (a, b) in [(enc[0], enc[1]), (enc[1], enc[2])]:
    arrow(ax, (a[2] + 0.9, a[3]), (b[2] + 0.9, b[3] + 0.8), 'MaxPool', color=RED[1], off=(-0.55, 0), fs=8.5)
arrow(ax, (enc[2][2] + 0.9, enc[2][3]), (5.5, 1.4), 'MaxPool', color=RED[1], off=(-0.55, -0.1), fs=8.5)
arrow(ax, (7.5, 1.4), (dec[2][2] + 0.9, dec[2][3]), '转置卷积', color=ORANGE[1], off=(0.6, -0.1), fs=8.5)
for (a, b) in [(dec[2], dec[1]), (dec[1], dec[0])]:
    arrow(ax, (a[2] + 0.9, a[3] + 0.8), (b[2] + 0.9, b[3]), '转置卷积', color=ORANGE[1], off=(-0.75, 0.12), fs=8.5)
for e, d in zip(enc, dec):
    arrow(ax, (e[2] + 1.8, e[3] + 0.4), (d[2], d[3] + 0.4), '跳跃连接：在通道维度上拼接', color='#868e96', ls='--', fs=8.5, off=(0, 0.16))
box(ax, 10.5, 0.35, 1.8, 0.8, 'head（1x1 卷积）\n4 x 64 x 64', ORANGE, fs=9)
arrow(ax, (11.4, 4.9), (11.4, 1.15))
label(ax, 11.6, 3.0, '每个像素\n4 个类别\n的 logits', fs=9, color='#495057', ha='left')
label(ax, 3.6, 0.45, '方框内每一层都是两个 3x3 卷积（DoubleConv）', fs=9, color='#868e96')
save(fig, T, 'unet.png')

# ---- 4. training
H = json.load(open(W + 'history.json'))
loss = np.array(H['loss']); steps = len(loss) // len(H['miou'])
fig = plt.figure(figsize=(12.6, 8.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.6], hspace=0.28)
top = gs[0].subgridspec(1, 2, wspace=0.25)
ax = fig.add_subplot(top[0])
ax.plot(np.arange(1, len(loss) + 1), loss, color=BLUE[0].replace('#e7f0fd', '#9ec0f0'), lw=0.8, label='每一步')
k = 10; sm = np.convolve(loss, np.ones(k) / k, mode='valid')
ax.plot(np.arange(k, len(loss) + 1), sm, color=BLUE[1], lw=1.8, label='10 步滑动平均')
ax.set_yscale('log'); ax.set_xlabel('训练步数'); ax.set_ylabel('交叉熵 + Dice 损失'); ax.legend(fontsize=9)
ax.grid(color='#e9ecef'); ax.set_title('训练损失', fontsize=11)
ax = fig.add_subplot(top[1]); ep = np.arange(1, len(H['miou']) + 1)
ax.plot(ep, H['accuracy'], 'o-', color='#868e96', label='像素准确率')
ax.plot(ep, H['miou'], 's-', color=GREEN[1], label='mIoU')
for e, a, m in zip(ep, H['accuracy'], H['miou']):
    if e in (1, 2, 8): ax.annotate(f'{m:.4f}', (e, m), xytext={ 1: (7, 0), 2: (5, -12), 8: (-8, -14) }[e], textcoords='offset points', fontsize=8.5, color=GREEN[1], ha='right' if e == 8 else 'left')
ax.set_xlabel('epoch'); ax.set_ylim(0, 1.05); ax.legend(fontsize=9, loc='lower right'); ax.grid(color='#e9ecef'); ax.set_title('验证集上的指标', fontsize=11)
bot = gs[1].subgridspec(3, 6, wspace=0.06, hspace=0.12)
for i in range(6):
    for r, (data, name) in enumerate([(D['p_img'], '图像'), (D['p_mask'], '标签'), (D['p_pred'], '预测')]):
        a = fig.add_subplot(bot[r, i])
        if r == 0: a.imshow(data[i])
        else: a.imshow(data[i], cmap=cmap, norm=norm, interpolation='nearest')
        clean(a)
        if i == 0: a.set_ylabel(name, fontsize=10)
fig.legend(handles=[Patch(fc=c) for c in CC], labels=CLS, loc='lower center', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06)
save(fig, T, 'training.png')

# ---- 5. overlay
ov = mpimg.imread(W + 'overlay.png')
fig, ax = plt.subplots(figsize=(5.2, 5.4)); ax.imshow(ov, interpolation='nearest'); clean(ax)
ax.set_title('预测结果叠加在原图上：白色为轮廓，彩色框为外接框\n两个相邻的矩形连成了一个连通域', fontsize=10.5)
fig.tight_layout(); save(fig, T, 'overlay.png')

# ---- 6. pretrained
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].imshow(D['f_img']); axs[0].set_title('输入：flower.jpg', fontsize=10.5)
pm = np.zeros(D['f_pred'].shape + (3,)); pm[:] = [0.93, 0.94, 0.95]; pm[D['f_pred'] == 16] = np.array([46, 153, 80]) / 255
axs[1].imshow(pm); axs[1].set_title('argmax 结果：绿色为 pottedplant（13.04%）', fontsize=10.5)
im = axs[2].imshow(D['f_pp'], cmap='viridis', vmin=0, vmax=1); axs[2].set_title('pottedplant 类别的 softmax 概率', fontsize=10.5)
for a in axs: clean(a)
fig.colorbar(im, ax=axs[2], fraction=0.035, pad=0.02)
fig.tight_layout(); save(fig, T, 'pretrained.png')
