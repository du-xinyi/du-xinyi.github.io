import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
T = '2026-11-11-pytorch-debugging'
D = json.load(open(HERE + '/data/pytorch-debugging/data.json'))
CB, CG, CO, CR, CP = BLUE[1], GREEN[1], ORANGE[1], RED[1], PURPLE[1]

def style(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(True, color='#e9ecef', lw=0.8); ax.set_axisbelow(True)

# ---- 1. flow
fig, ax = canvas(13.6, 9.4, 0.8)
steps = [('① 查看一个批次的数据', '形状、类型、数值范围，标签与样本是否对应', BLUE),
         ('② 检查模型的输出', '输出形状，初始损失是否接近 ln K', BLUE),
         ('③ 检查损失函数', '输入是否为 logits，两个输入的形状和类型', BLUE),
         ('④ 检查优化器与学习率', '参数是否交给了优化器，每一步是否清零梯度', BLUE),
         ('⑤ 在一个批次上过拟合', '同一批数据反复训练，损失应当降到接近 0', ORANGE),
         ('⑥ 完整训练', '同时记录训练和验证的损失曲线', GREEN)]
x, w, h, gap, top = 2.6, 4.3, 0.74, 0.42, 9.0
ys = []
for i, (t, note, c) in enumerate(steps):
    y = top - i * (h + gap) - h
    box(ax, x, y, w, h, t, c, fs=10.5, bold=(i == 4)); ys.append(y)
    label(ax, x + w + 0.3, y + h / 2, note, 9.2, '#868e96', ha='left')
for a, b in zip(ys, ys[1:]): arrow(ax, (x + w / 2, a), (x + w / 2, b + h), lw=1.2)
label(ax, x + w / 2 + 0.2, ys[4] - gap / 2, '能', 9.2, CG, ha='left')
# loop back
lx = 1.5
ax.plot([x, lx, lx], [ys[4] + h / 2, ys[4] + h / 2, ys[0] + h / 2], color=CR, lw=1.3)
arrow(ax, (lx, ys[0] + h / 2), (x, ys[0] + h / 2), color=CR, lw=1.3)
ax.text(lx - 0.25, (ys[0] + ys[4]) / 2 + h / 2, '不能：代码有错误，回到前面逐项检查', rotation=90, ha='center', va='center', fontsize=9.2, color=CR)
# outcomes
oy = 0.25; ow = 3.9
outs = [('损失变成 NaN', '定位第一个出现非有限值的位置', RED), ('训练损失降不下去', '学习率、梯度与激活值、模型容量', PURPLE), ('训练集好，验证集差', '过拟合：数据增强、正则化、早停', PURPLE)]
for i, (t, note, c) in enumerate(outs):
    ox = 0.5 + i * (ow + 0.45)
    box(ax, ox, oy, ow, 1.05, '', c)
    label(ax, ox + ow / 2, oy + 0.72, t, 10.2, bold=True); label(ax, ox + ow / 2, oy + 0.32, note, 8.8, '#495057')
    arrow(ax, (x + w / 2, ys[5]), (ox + ow / 2, oy + 1.05), lw=1.1)
save(fig, T, 'flow.png')

# ---- 2. overfit one batch
fig, ax = plt.subplots(figsize=(8.2, 4.3))
for l, name, c in zip(D['overfit'], ['正确：损失降到接近 0', '优化器绑定了旧模型的参数：损失完全不变', '输出多了一个 softmax：损失停在 1.46 以上'], [CG, CR, CO]):
    ax.plot(range(1, len(l) + 1), l, color=c, lw=2, label=name)
ax.axhline(np.log(10), color='#adb5bd', lw=1, ls='--'); ax.text(200, np.log(10) + 0.05, 'ln 10 ≈ 2.30', ha='right', va='bottom', fontsize=9, color='#868e96')
ax.axhline(1.4612, color='#adb5bd', lw=1, ls=':'); ax.text(200, 1.4612 - 0.05, '1.46', ha='right', va='top', fontsize=9, color='#868e96')
ax.set_xlabel('训练步数'); ax.set_ylabel('损失'); ax.set_xlim(0, 200); ax.set_ylim(-0.05, 2.75)
ax.set_title('在同一个批次（32 个样本）上反复训练', fontsize=11.5); ax.legend(frameon=False, fontsize=9.5, loc=(0.36, 0.22)); style(ax)
fig.tight_layout(); save(fig, T, 'overfit_one_batch.png')

# ---- 3. learning rate
fig, ax = plt.subplots(figsize=(8.2, 4.3))
for lr, name, c in [('1.0', '学习率 1.0（过大）：震荡，最后停在 ln 10', CR), ('0.1', '学习率 0.1（合适）：平稳下降', CG), ('0.001', '学习率 0.001（过小）：几乎不下降', CB)]:
    l = D['lr'][lr]; ax.plot(range(1, len(l) + 1), l, color=c, lw=2, marker='o', ms=3, label=name)
ax.set_yscale('log'); ax.set_xlabel('轮数'); ax.set_ylabel('训练损失（对数坐标）'); ax.set_xlim(0.5, 30.5)
ax.set_title('同一个模型使用三种学习率（SGD）', fontsize=11.5); ax.legend(frameon=False, fontsize=9.5, loc='lower left'); style(ax)
fig.tight_layout(); save(fig, T, 'learning_rate.png')

# ---- 4. gradient norms
fig, ax = plt.subplots(figsize=(8.2, 4.3))
for name, lab, c in [('Sigmoid', 'Sigmoid：越靠近输入梯度越小（梯度消失）', CR), ('ReLU', 'ReLU，默认初始化', CO), ('ReLU + Kaiming', 'ReLU，Kaiming 初始化：各层接近', CG)]:
    l = D['grad'][name]; ax.plot(range(1, len(l) + 1), l, color=c, lw=2, marker='o', ms=5, label=lab)
ax.set_yscale('log'); ax.set_xticks(range(1, 12)); ax.set_xlabel('线性层的序号（1 最靠近输入，11 为输出层）'); ax.set_ylabel('权重梯度的范数（对数坐标）')
ax.set_title('10 个隐藏层的全连接网络，一次反向传播后各层的梯度', fontsize=11.5); ax.legend(frameon=False, fontsize=9.5, loc='lower right'); style(ax)
fig.tight_layout(); save(fig, T, 'gradient_norm.png')

# ---- 5. fit
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
for ax, key, title in zip(axs, ['under', 'over', 'ok'], ['欠拟合：两条曲线都停在较高的位置', '过拟合：训练损失趋于 0，验证损失回升', '正常：两条曲线都下降并保持接近']):
    h = np.array(D['fit'][key]); e = np.arange(1, len(h) + 1)
    ax.plot(e, h[:, 0], color=CB, lw=2, label='训练损失'); ax.plot(e, h[:, 1], color=CO, lw=2, label='验证损失')
    if key == 'over':
        b = int(h[:, 1].argmin()); ax.axvline(b + 1, color='#adb5bd', lw=1, ls='--')
        ax.text(b + 8, 2.1, f'验证损失最低\n（第 {b + 1} 轮）', fontsize=9, color='#868e96', va='center')
    ax.set_ylim(0, 2.4); ax.set_xlim(0, 300); ax.set_xlabel('轮数'); ax.set_title(title, fontsize=10.5); style(ax)
axs[0].set_ylabel('损失'); axs[0].legend(frameon=False, fontsize=9.5, loc='lower left')
fig.tight_layout(w_pad=1.5); save(fig, T, 'fit.png')
