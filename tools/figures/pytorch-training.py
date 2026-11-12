import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
T = '2026-11-11-pytorch-training'
W = HERE + '/data/pytorch-training/'
COLORS = ['#2f6fd0', '#d9822b', '#2e9950', '#7a4fd0']

def style(ax):
    ax.grid(True, color='#e9ecef', lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#adb5bd')
    ax.tick_params(colors='#495057')

# ---- 1. workflow
fig, ax = canvas(16.0, 6.5, 0.8)
label(ax, 5.2, 6.15, '训练：每个批次', 12, bold=True)
xs = [0.4, 2.9, 5.4, 7.9]
names = [('取一个批次\nDataLoader', GREY), ('前向传播\nmodel(x)', BLUE), ('计算损失\nloss', ORANGE), ('清空梯度\nzero_grad()', GREY)]
for x, (t, c) in zip(xs, names): box(ax, x, 4.4, 2.1, 1.0, t, c, fs=9.6)
for a, b in zip(xs, xs[1:]): arrow(ax, (a + 2.1, 4.9), (b, 4.9), lw=1.2)
xs2 = [7.9, 5.4, 2.9]
names2 = [('反向传播\nloss.backward()', GREEN), ('更新参数\noptimizer.step()', GREEN), ('更新学习率\nscheduler.step()', PURPLE)]
for x, (t, c) in zip(xs2, names2): box(ax, x, 2.6, 2.1, 1.0, t, c, fs=9.6)
arrow(ax, (8.95, 4.4), (8.95, 3.6), lw=1.2)
arrow(ax, (7.9, 3.1), (7.5, 3.1), lw=1.2); arrow(ax, (5.4, 3.1), (5.0, 3.1), lw=1.2)
ax.plot([2.9, 1.45], [3.1, 3.1], color='#495057', lw=1.2); arrow(ax, (1.45, 3.1), (1.45, 4.4), '下一个批次', lw=1.2, off=(0.62, 0.0), fs=8.8)
box(ax, 0.15, 2.3, 10.1, 3.4, '', ('none', '#adb5bd'), lw=1.1, ls='--', round=0.15)
label(ax, 0.4, 2.05, 'model.train()', 9, '#868e96', ha='left', mono=True)
label(ax, 13.9, 6.15, '每一轮结束后', 12, bold=True)
items = [('在验证集上评估\nmodel.eval() + no_grad()', BLUE), ('指标更好时\n保存最佳模型', ORANGE), ('保存检查点\n用于中断后恢复', GREY), ('判断是否提前停止', PURPLE)]
y = 5.6
for i, (t, c) in enumerate(items):
    h = 0.95 if '\n' in t else 0.7
    box(ax, 12.2, y - h, 3.4, h, t, c, fs=9.6)
    if i < len(items) - 1: arrow(ax, (13.9, y - h), (13.9, y - h - 0.3), lw=1.2)
    y -= h + 0.3
arrow(ax, (10.25, 5.12), (12.2, 5.12), '一轮结束', lw=1.2, off=(0, 0.22), fs=8.8)
box(ax, 0.4, 0.45, 9.6, 0.95, '训练前：划分数据集 → 创建模型、优化器、调度器 → 固定随机种子\n训练后：加载最佳模型，在测试集上评估一次', GREY, fs=9.4)
save(fig, T, 'workflow.png')

# ---- 2. augmentation
orig = np.load(W + 'aug_orig.npy'); tr = np.load(W + 'aug_train.npy'); ev = np.load(W + 'aug_eval.npy')
fig = plt.figure(figsize=(9.6, 3.9))
gs = fig.add_gridspec(2, 5, width_ratios=[1.5, 1, 1, 1, 1], wspace=0.08, hspace=0.3, left=0.02, right=0.98, top=0.9, bottom=0.03)
ax = fig.add_subplot(gs[:, 0]); ax.imshow(orig); ax.set_title('原图 640×427', fontsize=10.5); ax.axis('off')
for i in range(4):
    ax = fig.add_subplot(gs[0, i + 1]); ax.imshow(tr[i]); ax.axis('off')
    if i == 0: ax.set_title('训练变换：每次结果不同', fontsize=10.5, loc='left')
ax = fig.add_subplot(gs[1, 1]); ax.imshow(ev); ax.axis('off'); ax.set_title('验证变换：结果固定', fontsize=10.5, loc='left')
ax = fig.add_subplot(gs[1, 2:]); ax.axis('off')
ax.text(0.06, 0.5, '训练：随机裁剪缩放 + 随机翻转 + 颜色抖动\n验证：缩放短边到 256 + 中心裁剪 224\n两者的输出大小和归一化完全相同', fontsize=10, color='#495057', va='center', linespacing=1.7)
save(fig, T, 'augmentation.png')

# ---- 3. optimizers
res = json.load(open(W + 'optim_curves.json'))
fig, ax = plt.subplots(figsize=(8.6, 4.2))
k = 11
for (name, v), c in zip(res.items(), COLORS):
    v = np.array(v); sm = np.convolve(v, np.ones(k) / k, mode='valid')
    ax.plot(np.arange(len(sm)) + k // 2, sm, color=c, lw=1.8, ls='--' if name == 'AdamW' else '-', label={'SGD': 'SGD，lr=0.05', 'SGD + momentum 0.9': 'SGD，lr=0.05，momentum=0.9', 'Adam': 'Adam，lr=0.001', 'AdamW': 'AdamW，lr=0.001'}[name])
ax.set_yscale('log'); ax.set_xlabel('训练步数'); ax.set_ylabel('训练损失（对数坐标，11 步滑动平均）')
ax.set_title('同一个模型在手写数字数据集上使用不同优化器的训练损失', fontsize=11.5)
style(ax); ax.legend(frameon=False, fontsize=9.5); fig.tight_layout(); save(fig, T, 'optimizers.png')

# ---- 4. lr schedules
cur = json.load(open(W + 'lr_curves.json'))
fig, ax = plt.subplots(figsize=(8.6, 4.0))
for (name, v), c in zip(cur.items(), COLORS):
    ax.plot(v, color=c, lw=2, label={'warmup_cosine': '线性预热 + 余弦退火', 'one_cycle': 'OneCycleLR', 'step_decay': 'MultiStepLR（阶梯衰减）'}[name])
ax.set_xlabel('训练步数（共 30 轮，每轮 22 步）'); ax.set_ylabel('学习率'); ax.set_title('三种调度器实际产生的学习率', fontsize=11.5)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
style(ax); ax.legend(frameon=False, fontsize=9.5); fig.tight_layout(); save(fig, T, 'lr_schedules.png')

# ---- 5. training curves
h = np.array(json.load(open(W + 'history.json'))); ep = np.arange(1, len(h) + 1); best = int(h[:, 3].argmax())
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0))
axs[0].plot(ep, h[:, 0], color=COLORS[0], lw=2, label='训练集'); axs[0].plot(ep, h[:, 2], color=COLORS[1], lw=2, label='验证集')
axs[0].set_yscale('log'); axs[0].set_ylabel('损失（对数坐标）'); axs[0].set_title('损失', fontsize=11.5)
axs[1].plot(ep, h[:, 1], color=COLORS[0], lw=2, label='训练集'); axs[1].plot(ep, h[:, 3], color=COLORS[1], lw=2, label='验证集')
axs[1].scatter([ep[best]], [h[best, 3]], color='#d64545', zorder=5, s=36)
axs[1].annotate(f'最佳：第 {ep[best]} 轮，{h[best, 3]:.4f}', (ep[best], h[best, 3]), (ep[best] - 2.5, 0.8), fontsize=9.5, color='#d64545', ha='center',
                arrowprops=dict(arrowstyle='-', color='#d64545', lw=1))
axs[1].set_ylim(0.6, 1.01); axs[1].set_ylabel('准确率'); axs[1].set_title('准确率（纵轴从 0.6 开始）', fontsize=11.5)
for ax in axs: ax.set_xlabel('轮数'); style(ax); ax.legend(frameon=False, fontsize=9.5, loc='center right')
fig.tight_layout(w_pad=2.5); save(fig, T, 'curves.png')
