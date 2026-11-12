import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
T = '2026-11-11-evaluation-metrics'
D = json.load(open(HERE + '/data/evaluation-metrics/figdata.json'))
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY = '#2f6fd0', '#d9822b', '#2e9950', '#d64545', '#868e96'

def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#adb5bd')
    ax.tick_params(colors='#495057', labelsize=9.5); ax.grid(color='#e9ecef', lw=0.8); ax.set_axisbelow(True)

# ---- binary confusion matrix diagram
fig, ax = canvas(11.2, 5.9, 0.85)
x0, y0, w, h = 3.0, 1.5, 2.6, 1.5
label(ax, x0 + w, y0 + 2 * h + 0.95, '模型的预测', 11, bold=True)
label(ax, x0 + w / 2, y0 + 2 * h + 0.4, '预测为正类', 10.5); label(ax, x0 + 1.5 * w, y0 + 2 * h + 0.4, '预测为负类', 10.5)
label(ax, 0.75, y0 + h, '真实\n标签', 11, bold=True)
label(ax, 2.1, y0 + 1.5 * h, '实际为正类', 10.5); label(ax, 2.1, y0 + 0.5 * h, '实际为负类', 10.5)
cells = [(0, 1, 'TP 真正例', '正类，预测正确', GREEN), (1, 1, 'FN 假负例', '正类被漏掉（漏报）', RED),
         (0, 0, 'FP 假正例', '负类被当成正类（误报）', ORANGE), (1, 0, 'TN 真负例', '负类，预测正确', GREY)]
for cx, cy, a, b, c in cells:
    box(ax, x0 + cx * w + 0.06, y0 + cy * h + 0.06, w - 0.12, h - 0.12, '', c)
    label(ax, x0 + cx * w + w / 2, y0 + cy * h + h / 2 + 0.25, a, 12, bold=True)
    label(ax, x0 + cx * w + w / 2, y0 + cy * h + h / 2 - 0.3, b, 9.3, '#495057')
# recall: row bracket on the right
xr = x0 + 2 * w + 0.2
ax.plot([xr, xr + 0.18, xr + 0.18, xr], [y0 + h + 0.1, y0 + h + 0.1, y0 + 2 * h - 0.1, y0 + 2 * h - 0.1], color=C_BLUE, lw=1.6)
label(ax, xr + 0.35, y0 + 1.5 * h + 0.22, '召回率 = TP / (TP + FN)', 10.5, C_BLUE, ha='left', bold=True)
label(ax, xr + 0.35, y0 + 1.5 * h - 0.25, '只看这一行：\n所有正类中找出了多少', 9.3, '#495057', ha='left')
# precision: column bracket at bottom
yb = y0 - 0.2
ax.plot([x0 + 0.1, x0 + 0.1, x0 + w - 0.1, x0 + w - 0.1], [yb, yb - 0.18, yb - 0.18, yb], color=C_GREEN, lw=1.6)
label(ax, x0 + 0.1, yb - 0.5, '精确率 = TP / (TP + FP)', 10.5, C_GREEN, ha='left', bold=True)
label(ax, x0 + 0.1, yb - 0.9, '只看这一列：预测为正类的有多少是对的', 9.3, '#495057', ha='left')
save(fig, T, 'confusion_binary.png')

# ---- digits confusion heatmap
from matplotlib.colors import LinearSegmentedColormap
m = np.array(D['matrix'])
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cfe0fa', '#2f6fd0', '#173f80'])
fig, ax = plt.subplots(figsize=(6.6, 5.9))
im = ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max())
for i in range(10):
    for j in range(10):
        v = m[i, j]
        if v == 0: ax.text(j, i, '0', ha='center', va='center', fontsize=9, color='#ced4da')
        elif i == j: ax.text(j, i, v, ha='center', va='center', fontsize=10, color='white')
        else: ax.text(j, i, v, ha='center', va='center', fontsize=10, color=C_RED if v >= 4 else INK, fontweight='bold' if v >= 4 else 'normal')
ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.tick_params(length=0, labelsize=10)
ax.set_xticks(np.arange(-.5, 10), minor=True); ax.set_yticks(np.arange(-.5, 10), minor=True)
ax.grid(which='minor', color='#e9ecef', lw=0.8); ax.tick_params(which='minor', length=0)
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
ax.set_xlabel('预测类别', fontsize=11, labelpad=8); ax.set_ylabel('真实类别', fontsize=11, labelpad=8)
for s in ax.spines.values(): s.set_color('#ced4da')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9, length=0); cb.set_label('样本数', fontsize=10)
ax.set_title('对角线为预测正确的样本，其余为各种错误（红色为 4 次及以上）', fontsize=10.5, color='#495057', y=-0.09)
fig.tight_layout(); save(fig, T, 'confusion_digits.png')

# ---- curves helpers
score = np.array(D['score']); target = np.array(D['target'])
def curves(s, t):
    o = np.argsort(-s, kind='stable'); t = t[o]; s = s[o]
    tp = np.cumsum(t); fp = np.cumsum(1 - t)
    return fp / fp[-1], tp / tp[-1], tp / (tp + fp), s
def auc_ap(s, t):
    f, r, p, _ = curves(s, t)
    return np.trapz(np.r_[0, r], np.r_[0, f]), np.sum((r - np.r_[0, r[:-1]]) * p)

# ---- threshold
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.2))
ax = axs[0]; clean(ax)
bins = np.linspace(0, 1, 41)
ax.hist(score[target == 0], bins, color=C_GREY, alpha=0.55, label='负类（8932 个）')
ax.hist(score[target == 1], bins, color=C_BLUE, alpha=0.75, label='正类（1068 个）')
ax.set_yscale('log'); ax.axvline(0.5, color=C_RED, lw=1.5, ls='--')
ax.text(0.515, 400, '阈值 0.5\n右侧预测为正类', color=C_RED, fontsize=9.5, va='top')
ax.set_xlabel('模型输出的正类概率', fontsize=10.5); ax.set_ylabel('样本数（对数坐标）', fontsize=10.5); ax.set_xlim(0, 1)
ax.legend(frameon=False, fontsize=9.5, loc='upper right'); ax.set_title('两类样本的得分分布', fontsize=11)
ax = axs[1]; clean(ax)
ths = np.linspace(0.01, 0.99, 197); P = []; R = []; F = []
for th in ths:
    pr = score >= th; tp = (pr & (target == 1)).sum(); fp = (pr & (target == 0)).sum(); fn = (~pr & (target == 1)).sum()
    P.append(tp / (tp + fp)); R.append(tp / (tp + fn)); F.append(2 * tp / (2 * tp + fp + fn))
ax.plot(ths, P, color=C_GREEN, lw=2, label='精确率'); ax.plot(ths, R, color=C_BLUE, lw=2, label='召回率'); ax.plot(ths, F, color=C_ORANGE, lw=2, label='F1')
k = int(np.argmax(F)); ax.plot(ths[k], F[k], 'o', color=C_ORANGE, ms=6)
ax.annotate(f'F1 最高 {F[k]:.3f}\n阈值 {ths[k]:.2f}', (ths[k], F[k]), (ths[k] + 0.02, F[k] - 0.3), fontsize=9.5, color=C_ORANGE,
            arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=1))
ax.axvline(0.5, color=C_RED, lw=1.2, ls='--')
ax.set_xlabel('阈值', fontsize=10.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=9.5, loc='lower center', ncol=3)
ax.set_title('阈值越高，精确率越高、召回率越低', fontsize=11)
fig.tight_layout(w_pad=2.5); save(fig, T, 'threshold.png')
print('best f1', ths[k], F[k])

# ---- roc / pr
neg = score[target == 0]; sub = np.array(D['sub100'])
s2 = np.r_[sub, neg]; t2 = np.r_[np.ones(len(sub), int), np.zeros(len(neg), int)]
sets = [(score, target, C_BLUE, '正类占 10.68%'), (s2, t2, C_ORANGE, '正类占 1.11%')]
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.7))
for ax in axs: clean(ax); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.03); ax.set_aspect('equal')
for s, t, c, name in sets:
    f, r, p, _ = curves(s, t); a, ap = auc_ap(s, t); print(name, a, ap)
    axs[0].plot(np.r_[0, f], np.r_[0, r], color=c, lw=2, label=f'{name}，AUC = {a:.3f}')
    axs[1].plot(r, p, color=c, lw=2, label=f'{name}，AP = {ap:.3f}')
    axs[1].axhline(t.mean(), color=c, lw=1, ls=':')
axs[0].plot([0, 1], [0, 1], color=C_GREY, lw=1, ls=':'); axs[0].text(0.62, 0.55, '随机猜测', color=C_GREY, fontsize=9.5, rotation=45, ha='center', va='center')
axs[1].text(0.02, 0.135, '虚线：随机猜测的精确率，等于正类占比', color=C_GREY, fontsize=9.3)
axs[0].set_xlabel('假正率 FPR', fontsize=10.5); axs[0].set_ylabel('真正率 TPR（召回率）', fontsize=10.5); axs[0].set_title('ROC 曲线：几乎不受正类占比影响', fontsize=11)
axs[1].set_xlabel('召回率', fontsize=10.5); axs[1].set_ylabel('精确率', fontsize=10.5); axs[1].set_title('PR 曲线：正类越少，曲线越低', fontsize=11)
axs[0].legend(frameon=False, fontsize=9.5, loc='lower right'); axs[1].legend(frameon=False, fontsize=9.5, loc='lower left', bbox_to_anchor=(0.0, 0.17))
fig.tight_layout(w_pad=2.5); save(fig, T, 'roc_pr.png')

# ---- iou / dice
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw=dict(width_ratios=[1, 1.05]))
ax = axs[0]; ax.set_xlim(0, 100); ax.set_ylim(100, 0); ax.set_aspect('equal')
ax.add_patch(Rectangle((20, 20), 40, 40, fc='#e7f0fd', ec=C_BLUE, lw=1.8)); ax.add_patch(Rectangle((30, 30), 40, 40, fc='#fdf0e1', ec=C_ORANGE, lw=1.8, alpha=0.9))
ax.add_patch(Rectangle((30, 30), 30, 30, fc='#e6f6ea', ec=C_GREEN, lw=1.8))
ax.text(25, 25, '真实', color=C_BLUE, fontsize=10.5, ha='center', va='center'); ax.text(65, 65, '预测', color=C_ORANGE, fontsize=10.5, ha='center', va='center')
ax.text(45, 45, '交集\n900', color=C_GREEN, fontsize=10.5, ha='center', va='center', fontweight='bold')
ax.text(50, 80, '并集 = 1600 + 1600 − 900 = 2300', fontsize=10, ha='center', color=INK)
ax.text(50, 88, 'IoU = 900 / 2300 = 0.391        Dice = 1800 / 3200 = 0.563', fontsize=10, ha='center', color=INK)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color('#ced4da')
ax.set_title('两个 40×40 的掩码相互偏移 10 个像素', fontsize=11)
ax = axs[1]; clean(ax)
x = np.linspace(0, 1, 200); ax.plot(x, 2 * x / (1 + x), color=C_ORANGE, lw=2.2, label='Dice = 2·IoU / (1 + IoU)'); ax.plot(x, x, color=C_GREY, lw=1, ls=':', label='Dice = IoU')
ax.plot(0.3913, 0.5625, 'o', color=C_GREEN, ms=7); ax.annotate('左图的例子\n(0.391, 0.563)', (0.3913, 0.5625), (0.47, 0.36), fontsize=9.5, color=C_GREEN, arrowprops=dict(arrowstyle='-', color=C_GREEN, lw=1))
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect('equal'); ax.set_xlabel('IoU', fontsize=10.5); ax.set_ylabel('Dice', fontsize=10.5)
ax.legend(frameon=False, fontsize=9.5, loc='lower right'); ax.set_title('同一个预测，Dice 总是不低于 IoU', fontsize=11)
fig.tight_layout(w_pad=2); save(fig, T, 'iou_dice.png')
