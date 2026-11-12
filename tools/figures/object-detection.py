import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
T = '2026-11-11-object-detection'
W = HERE + '/data/object-detection/'
D = json.load(open(W + 'fig.json'))
MONO = 'Noto Sans Mono CJK SC'

def rect(ax, b, c, lw=2, fill=False, ls='-', alpha=1.0, z=2):
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc=c[0] if fill else 'none', ec=c[1], lw=lw, ls=ls, alpha=alpha, zorder=z))

def frame(ax, w, h, ticks=True):
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect('equal')
    for s in ax.spines.values(): s.set_color('#adb5bd')
    ax.tick_params(colors='#868e96', labelsize=8, length=2)
    if not ticks: ax.set_xticks([]); ax.set_yticks([])

def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

# ---- 1. box formats
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
b = [100, 120, 300, 360]
def dims(ax, x0, y0):
    ax.annotate('', (300, y0), (100, y0), arrowprops=dict(arrowstyle='<->', color=ORANGE[1], lw=1.3))
    ax.text(200, y0 + 10, 'w = 200', ha='center', va='top', fontsize=9, color=ORANGE[1])
    ax.annotate('', (x0, 360), (x0, 120), arrowprops=dict(arrowstyle='<->', color=ORANGE[1], lw=1.3))
    ax.text(x0 + 12, 240, 'h = 240', ha='left', va='center', fontsize=9, color=ORANGE[1])
for ax, name in zip(axs, ['xyxy', 'xywh', 'cxcywh']):
    frame(ax, 640, 480); ax.set_xticks([0, 160, 320, 480, 640]); ax.set_yticks([0, 120, 240, 360, 480])
    ax.xaxis.tick_top(); rect(ax, b, BLUE, fill=True)
    ax.set_xlabel('x（向右）', fontsize=9, color='#868e96'); ax.xaxis.set_label_position('top'); ax.set_ylabel('y（向下）', fontsize=9, color='#868e96')
    if name == 'xyxy':
        ax.plot([100, 300], [120, 360], 'o', color=RED[1], ms=6, zorder=5)
        ax.text(92, 108, '(x1, y1) = (100, 120)', fontsize=9, color=RED[1], ha='left', va='bottom')
        ax.text(308, 372, '(x2, y2) = (300, 360)', fontsize=9, color=RED[1], ha='left', va='top')
        t = '[100, 120, 300, 360]'
    elif name == 'xywh':
        ax.plot([100], [120], 'o', color=RED[1], ms=6, zorder=5)
        ax.text(92, 108, '(x, y) = (100, 120)', fontsize=9, color=RED[1], ha='left', va='bottom')
        dims(ax, 320, 385); t = '[100, 120, 200, 240]'
    else:
        ax.plot([200], [240], 'o', color=RED[1], ms=6, zorder=5)
        ax.text(200, 222, '(cx, cy)\n= (200, 240)', fontsize=9, color=RED[1], ha='center', va='bottom', linespacing=1.2)
        dims(ax, 320, 385); t = '[200, 240, 200, 240]'
    ax.set_title(f'{name}\n{t}', y=-0.24, fontsize=11)
fig.tight_layout(w_pad=2.5); save(fig, T, 'box_formats.png')

# ---- 2. iou cases
gt = [100, 100, 300, 300]
cases = [([100, 100, 300, 300], '完全重合'), ([120, 120, 320, 320], '偏移 20 像素'), ([150, 150, 350, 350], '偏移 50 像素'),
         ([200, 200, 400, 400], '偏移 100 像素'), ([150, 150, 250, 250], '在真实框内部'), ([320, 100, 420, 300], '不相交')]
fig, axs = plt.subplots(1, 6, figsize=(13.2, 2.9))
for ax, (p, name) in zip(axs, cases):
    frame(ax, 500, 500, ticks=False); ax.set_xlim(40, 460); ax.set_ylim(460, 40)
    for s in ax.spines.values(): s.set_visible(False)
    inter = [max(gt[0], p[0]), max(gt[1], p[1]), min(gt[2], p[2]), min(gt[3], p[3])]
    if inter[2] > inter[0] and inter[3] > inter[1]:
        ax.add_patch(Rectangle((inter[0], inter[1]), inter[2] - inter[0], inter[3] - inter[1], fc='#fbd9a8', ec='none', zorder=1))
    rect(ax, gt, GREEN, lw=2.2); rect(ax, p, BLUE, lw=2.2, ls='--')
    ax.set_title(f'IoU = {iou(gt, p):.4f}', fontsize=11.5, color=INK, pad=4)
    ax.text(250, 475, name, ha='center', va='top', fontsize=10, color='#495057')
    print(name, round(iou(gt, p), 4))
fig.legend(handles=[Rectangle((0, 0), 1, 1, fc='none', ec=GREEN[1], lw=2.2), Rectangle((0, 0), 1, 1, fc='none', ec=BLUE[1], lw=2.2, ls='--'), Rectangle((0, 0), 1, 1, fc='#fbd9a8', ec='none')],
           labels=['真实框', '预测框', '交集'], loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1)); save(fig, T, 'iou_cases.png')

# ---- 3. nms
boxes = [[100, 100, 300, 300], [110, 105, 310, 295], [90, 110, 290, 310], [400, 150, 560, 330], [410, 160, 570, 340], [380, 140, 540, 310], [250, 120, 450, 320]]
scores = [0.90, 0.75, 0.60, 0.85, 0.70, 0.55, 0.30]
cols = [BLUE, BLUE, BLUE, GREEN, GREEN, GREEN, PURPLE]
keep = [0, 3, 6]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.5))
for ax, idx, title in zip(axs, [range(7), keep], ['抑制前：7 个候选框', '抑制后（IoU 阈值 0.5）：保留 3 个']):
    frame(ax, 660, 400); ax.set_ylim(400, 50)
    for i in idx:
        rect(ax, boxes[i], cols[i], lw=2.4 if i in keep else 1.3, ls='-' if i in keep else '--')
    lab = { 0: (100, 97, 'left', 'bottom'), 1: (310, 97, 'right', 'bottom'), 2: (90, 313, 'left', 'top'), 3: (407, 157, 'left', 'top'),
            4: (570, 343, 'right', 'top'), 5: (540, 137, 'right', 'bottom'), 6: (350, 323, 'center', 'top') }
    for i in idx:
        x, y, ha, va = lab[i]
        ax.text(x, y, f'{scores[i]:.2f}', ha=ha, va=va, fontsize=9.5, color=cols[i][1], fontweight='bold' if i in keep else 'normal')
    ax.set_title(title, fontsize=11.5)
fig.tight_layout(w_pad=2.5); save(fig, T, 'nms.png')

# ---- 4. anchors vs points
gtb = [100, 80, 220, 200]
fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.9))
for ax in axs:
    frame(ax, 320, 320); ax.set_xticks(range(0, 321, 64)); ax.set_yticks(range(0, 321, 64))
    for v in range(0, 321, 16):
        ax.plot([v, v], [0, 320], color='#e9ecef', lw=0.7, zorder=0); ax.plot([0, 320], [v, v], color='#e9ecef', lw=0.7, zorder=0)
ax = axs[0]
ax.set_xlim(-40, 360); ax.set_ylim(360, -40)
cx, cy = 168, 136
for size, c in [(64, BLUE), (128, PURPLE)]:
    for r in [0.5, 1.0, 2.0]:
        w = size / np.sqrt(r); h = size * np.sqrt(r)
        rect(ax, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], c, lw=1.5)
rect(ax, gtb, GREEN, lw=2.6)
ax.plot([cx], [cy], 'o', color=RED[1], ms=5, zorder=6)
pos = np.array(D['anchor_pos']); pc = np.unique(np.stack([(pos[:, 0] + pos[:, 2]) / 2, (pos[:, 1] + pos[:, 3]) / 2], 1), axis=0)
ax.set_title(f'锚框：每个格子放 6 个预设框，共 2400 个\n与真实框 IoU 不低于 0.5 的 {len(pos)} 个为正样本', fontsize=11, pad=8)
ax.legend(handles=[Rectangle((0, 0), 1, 1, fc='none', ec=GREEN[1], lw=2.6), Rectangle((0, 0), 1, 1, fc='none', ec=BLUE[1], lw=1.5), Rectangle((0, 0), 1, 1, fc='none', ec=PURPLE[1], lw=1.5)],
          labels=['真实框', '边长 64 的锚框', '边长 128 的锚框'], loc='lower right', fontsize=8.5, frameon=True, framealpha=0.95)
ax = axs[1]
ax.set_xlim(-40, 360); ax.set_ylim(360, -40)
gx, gy = np.meshgrid(np.arange(20) * 16 + 8, np.arange(20) * 16 + 8)
ax.plot(gx.ravel(), gy.ravel(), '.', color='#ced4da', ms=3.5, zorder=1)
pin = np.array(D['points_in']); ax.plot(pin[:, 0], pin[:, 1], '.', color=ORANGE[1], ms=6, zorder=3)
rect(ax, gtb, GREEN, lw=2.6)
px, py = 136, 120
for (ex, ey) in [(gtb[0], py), (gtb[2], py), (px, gtb[1]), (px, gtb[3])]:
    ax.annotate('', (ex, ey), (px, py), arrowprops=dict(arrowstyle='-|>', color=RED[1], lw=1.5, shrinkA=0, shrinkB=0), zorder=5)
ax.plot([px], [py], 'o', color=RED[1], ms=5, zorder=6)
for t, x, y in [('l', 116, 113), ('r', 182, 113), ('t', 143, 98), ('b', 143, 164)]:
    ax.text(x, y, t, fontsize=10, color=RED[1], ha='center', va='center', fontweight='bold', zorder=7,
            bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
ax.set_title(f'无锚框：每个格子一个点，共 400 个\n落在真实框内的 {len(pin)} 个点为正样本，回归到四条边的距离', fontsize=11, pad=8)
fig.tight_layout(w_pad=3); save(fig, T, 'anchors.png')

# ---- 5. PR curve
rec, pre, env = np.array(D['recall']), np.array(D['precision']), np.array(D['envelope'])
fig, ax = plt.subplots(figsize=(7.6, 4.6))
xs = np.concatenate([[0], np.repeat(rec, 2)[:-1], [rec[-1]]]); prev = np.concatenate([[0], rec[:-1]])
sx = []; sy = []
for a, b_, e in zip(prev, rec, env):
    sx += [a, b_]; sy += [e, e]
ax.fill_between(sx, sy, color=BLUE[0], zorder=1, label=f'包络下的面积 AP = {D["ap"]:.4f}')
ax.plot(sx, sy, color=BLUE[1], lw=1.6, ls='--', zorder=2, label='包络（右侧精确率的最大值）')
ax.plot(rec, pre, '-', color='#868e96', lw=1.2, zorder=3)
tp = np.array(D['tp']).astype(bool)
ax.plot(rec[tp], pre[tp], 'o', color=GREEN[1], ms=7, zorder=4, label='TP')
ax.plot(rec[~tp], pre[~tp], 'X', color=RED[1], ms=7, zorder=4, label='FP')
offs = [(8, 6), (8, 6), (9, -2), (9, -2), (9, -3), (-2, -15), (8, 6), (9, -3)]
for r, p, s, o in zip(rec, pre, D['scores'], offs):
    ax.annotate(f'{s:.2f}', (r, p), xytext=o, textcoords='offset points', fontsize=8.5, color='#495057')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.08); ax.set_xlabel('召回率'); ax.set_ylabel('精确率')
ax.set_title('按分数从高到低逐个加入预测框，点旁的数字为该预测框的分数', fontsize=11)
ax.grid(color='#e9ecef', lw=0.8); ax.set_axisbelow(True); ax.legend(loc='lower left', fontsize=9, frameon=True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.tight_layout(); save(fig, T, 'pr_curve.png')

# ---- 6. letterbox
orig = np.load(W + 'orig.npy'); lb = np.load(W + 'lb.npy')
names = ['person', 'tie']; cs = [BLUE, ORANGE]
fig, axs = plt.subplots(1, 3, figsize=(12.4, 5.0), gridspec_kw=dict(width_ratios=[512, 600, 512]))
def show(ax, img, bxs, title, ls='-'):
    ax.imshow(img); h, w = img.shape[:2]; ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.tick_params(colors='#868e96', labelsize=8, length=2); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_color('#adb5bd')
    for raw, n, c in zip(bxs, names, cs):
        bb = [min(max(v, 2), lim - 2) for v, lim in zip(raw, [w, h, w, h])]
        rect(ax, bb, c, lw=2.2, ls=ls, z=4)
        ax.text(bb[0] + 0.012 * w, bb[1] + 0.012 * h, f'{n} [{", ".join(str(round(v)) for v in raw)}]', fontsize=8, color='white', va='top', ha='left', zorder=5,
                bbox=dict(fc=c[1], ec='none', pad=1.5))
    ax.set_title(title, y=-0.13, fontsize=11)
show(axs[0], orig, [], '原图 512 x 600')
axs[0].set_xticks([0, 128, 256, 384, 512])
show(axs[1], lb, D['lb_boxes'], '缩放 0.5333 倍并左右各填充 23、24 像素\n在 320 x 320 的输入上检测')
axs[1].set_xticks([0, 23, 160, 296, 320]); axs[1].set_yticks([0, 80, 160, 240, 320])
show(axs[2], orig, D['mapped'], '减去填充、除以缩放比例\n映射回原图')
axs[2].set_xticks([0, 128, 256, 384, 512])
fig.tight_layout(w_pad=2.0); save(fig, T, 'letterbox.png', dpi=110)
from PIL import Image
f = ROOT + T + '/letterbox.png'; Image.open(f).convert('RGB').quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(f, optimize=True)
print(os.path.getsize(f) // 1024, 'KB')
