import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
from PIL import Image
T = '2026-11-12-experiment-tracking'
W = HERE + '/data/experiment-tracking/'
D = json.load(open(W + 'fig_data.json'))
COLORS = ['#2f6fd0', '#d9822b', '#2e9950', '#7a4fd0', '#d64545']

def style(ax):
    ax.grid(True, color='#e9ecef', lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#adb5bd')
    ax.tick_params(colors='#495057')

# ---- 1. overview
fig, ax = canvas(15.6, 6.3, 0.8)
label(ax, 2.2, 5.9, '训练开始前写入', 11.5, bold=True)
label(ax, 7.8, 5.9, '一次实验 = 一个目录', 11.5, bold=True)
label(ax, 13.4, 5.9, '训练过程中、结束时写入', 11.5, bold=True)
left = [('超参数与配置', BLUE), ('代码版本\ngit 提交、未提交的修改', BLUE), ('随机种子', BLUE), ('运行环境与启动命令', BLUE)]
y = 5.4
for t, c in left:
    h = 0.95 if '\n' in t else 0.7
    box(ax, 0.3, y - h, 3.8, h, t, c, fs=9.6); arrow(ax, (4.1, y - h / 2), (5.3, y - h / 2), lw=1.2); y -= h + 0.28
right = [('日志', GREEN), ('每一步、每一轮的指标', GREEN), ('TensorBoard 事件文件', GREEN), ('检查点', ORANGE), ('最终结果', ORANGE)]
y = 5.4
for t, c in right:
    h = 0.7
    box(ax, 11.5, y - h, 3.8, h, t, c, fs=9.6); arrow(ax, (11.5, y - h / 2), (10.3, y - h / 2), lw=1.2); y -= h + 0.2
box(ax, 5.3, 0.95, 5.0, 4.45, '', GREY, round=0.12)
label(ax, 5.55, 5.05, 'runs/' + D['run'][:24] + '…/', 9.2, ha='left', mono=True, bold=True)
files = [('config.json', '配置'), ('meta.json', '代码版本、环境、命令'), ('diff.patch', '未提交的修改'), ('train.log', '日志'),
         ('metrics.jsonl', '每一轮的指标'), ('events.out.tfevents.*', 'TensorBoard'), ('best.pt  last.pt', '检查点'), ('result.json', '最终结果')]
for i, (f, d) in enumerate(files):
    yy = 4.55 - i * 0.46
    label(ax, 5.75, yy, f, 9, ha='left', mono=True); label(ax, 10.1, yy, d, 8.6, '#868e96', ha='right')
label(ax, 7.8, 0.45, '复现一个结果、比较两次实验，需要的信息都从这个目录里读', 9.6, '#868e96')
save(fig, T, 'overview.png')

# ---- 2. curves from metrics.jsonl
m = D['metrics']; ep = [r['epoch'] for r in m]
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.7))
axs[0].plot(ep, [r['train_loss'] for r in m], color=COLORS[0], lw=2, label='训练集')
axs[0].plot(ep, [r['val_loss'] for r in m], color=COLORS[1], lw=2, label='验证集')
axs[0].set_yscale('log'); axs[0].set_xlabel('轮数'); axs[0].set_ylabel('损失'); axs[0].legend(frameon=False)
axs[1].plot(ep, [r['train_accuracy'] for r in m], color=COLORS[0], lw=2, label='训练集')
axs[1].plot(ep, [r['val_accuracy'] for r in m], color=COLORS[1], lw=2, label='验证集')
axs[1].set_ylim(0.9, 1.003); axs[1].set_xlabel('轮数'); axs[1].set_ylabel('准确率（只显示 0.9 以上）'); axs[1].legend(frameon=False, loc='lower right')
for a in axs: style(a)
fig.suptitle('由 metrics.jsonl 重新画出的曲线', fontsize=11.5, color='#495057', y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.5); save(fig, T, 'metrics_curves.png')

# ---- 3. read back from event file
sc = D['scalars']
fig, axs = plt.subplots(2, 2, figsize=(10.6, 7.2))
a = axs[0, 0]; x, v = zip(*sc['step/loss']); a.plot(x, v, color=COLORS[0], lw=1.6, label='step/loss')
a.set_yscale('log'); a.set_xlabel('全局步数'); a.set_ylabel('step/loss', color=COLORS[0]); style(a)
b = a.twinx(); x, v = zip(*sc['step/lr']); b.plot(x, v, color=COLORS[3], lw=1.6, ls='--')
b.set_ylabel('step/lr', color=COLORS[3]); b.spines['top'].set_visible(False); b.tick_params(colors='#495057')
b.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
a.set_title('add_scalar：每 10 步记录的损失与学习率', fontsize=10.5, color='#495057')
a = axs[0, 1]
for tag, c, n in [('accuracy/train', COLORS[0], 'accuracy/train'), ('accuracy/val', COLORS[1], 'accuracy/val')]:
    x, v = zip(*sc[tag]); a.plot(x, v, color=c, lw=1.8, marker='o', ms=2.8, label=n)
a.set_ylim(0.9, 1.003); a.set_xlabel('全局步数'); a.legend(frameon=False, loc='lower right', prop={'family': 'Noto Sans Mono CJK SC', 'size': 9}); style(a)
a.set_title('add_scalar：每一轮记录的准确率（只显示 0.9 以上）', fontsize=10.5, color='#495057')
a = axs[1, 0]; hs = D['hist']['weights/10.weight']
lo = min(h['min'] for h in hs); hi = max(h['max'] for h in hs); edges = np.linspace(lo, hi, 31); cen = (edges[:-1] + edges[1:]) / 2
for i, h in enumerate(hs):
    lim = np.array(h['limits']); cnt = np.array(h['counts']); left_e = np.concatenate([[h['min']], lim[:-1]])
    mid = np.clip((np.clip(left_e, h['min'], h['max']) + np.clip(lim, h['min'], h['max'])) / 2, lo, hi)
    dens, _ = np.histogram(mid, bins=edges, weights=cnt); dens = dens / dens.max() * 0.9
    base = i
    col = plt.cm.Blues(0.35 + 0.6 * i / (len(hs) - 1))
    a.fill_between(cen, base, base + dens, color=col, alpha=0.75, lw=0); a.plot(cen, base + dens, color=col, lw=1.1)
a.set_yticks(range(len(hs))); a.set_yticklabels([str(h['step']) for h in hs]); a.set_ylabel('全局步数'); a.set_xlabel('权重的取值'); style(a)
a.set_title('add_histogram：weights/10.weight 的分布', fontsize=10.5, color='#495057')
a = axs[1, 1]; img = Image.open(W + 'inputs.png'); a.imshow(img, interpolation='nearest'); a.axis('off')
a.set_title('add_image：inputs（make_grid 拼出的 32 张输入图像）', fontsize=10.5, color='#495057')
fig.suptitle('从事件文件中读回的数据，用 matplotlib 绘制（不是 TensorBoard 的界面截图）', fontsize=11.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.2, h_pad=2.0); save(fig, T, 'event_readback.png')

# ---- 4. sweep
sw = D['sweep']; lrs = sorted({r['lr'] for r in sw})
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.1))
a = axs[0]
for c, lr in zip(COLORS, lrs):
    acc = np.array([[row['val_accuracy'] for row in r['metrics']] for r in sw if r['lr'] == lr])
    e = np.arange(1, acc.shape[1] + 1); mu = acc.mean(0)
    a.fill_between(e, acc.min(0), acc.max(0), color=c, alpha=0.16, lw=0); a.plot(e, mu, color=c, lw=1.9, label=f'lr = {lr:g}')
a.set_ylim(0.9, 1.003); a.set_xlabel('轮数'); a.set_ylabel('验证集准确率（只显示 0.9 以上）'); a.legend(frameon=False, loc='lower right', fontsize=9); style(a)
a.set_title('5 个种子的平均值，阴影为最小值到最大值', fontsize=10.5, color='#495057')
a = axs[1]
for i, (c, lr) in enumerate(zip(COLORS, lrs)):
    v = np.array([r['final_val_accuracy'] for r in sw if r['lr'] == lr])
    a.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v, color=c, s=22, alpha=0.55, zorder=3, lw=0)
    a.errorbar(i + 0.3, v.mean(), yerr=v.std(ddof=1), color=c, marker='D', ms=6, capsize=4, lw=1.6, zorder=4)
a.set_xticks(np.arange(len(lrs)) + 0.15); a.set_xticklabels([f'{lr:g}' for lr in lrs]); a.set_xlabel('学习率'); a.set_ylabel('最后一轮的验证集准确率'); style(a)
a.set_title('圆点为各个种子，菱形为平均值 ± 标准差', fontsize=10.5, color='#495057')
fig.tight_layout(w_pad=2.5); save(fig, T, 'sweep.png')
