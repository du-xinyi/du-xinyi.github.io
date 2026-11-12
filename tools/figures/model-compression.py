import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json
from lib import *
import numpy as np
T = '2026-11-12-model-compression'
D = json.load(open(HERE + '/data/model-compression/results.json'))
which = sys.argv[1:] or ['pruning', 'sparsity', 'distillation', 'fake_quant', 'bits']

def tidy(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

# ---- 1. pruning：非结构化与结构化剪枝的示意
if 'pruning' in which:
    fig, ax = canvas(17.8, 5.5, 0.78)
    rng = np.random.RandomState(3); c = 0.42

    def grid(x0, y0, rows, cols, dead_rows=(), dead_cols=(), dead=None, color=BLUE):
        for r in range(rows):
            for k in range(cols):
                off = r in dead_rows or k in dead_cols or (dead is not None and dead[r, k])
                ax.add_patch(Rectangle((x0 + k * c, y0 + (rows - 1 - r) * c), c, c, fc='white' if off else color[0],
                                       ec='#ced4da' if off else color[1], lw=0.9, zorder=1 if off else 2))

    label(ax, 2.58, 5.1, '非结构化剪枝', 12, bold=True)
    grid(0.9, 1.75, 6, 8, dead=rng.rand(6, 8) < 0.6)
    label(ax, 0.6, 3.01, '输\n出\n通\n道', 9, '#868e96'); label(ax, 2.58, 4.5, '输入通道', 9, '#868e96')
    label(ax, 2.58, 1.15, '零散的权重被置零，矩阵的形状不变', 9.4)
    label(ax, 2.58, 0.45, '文件大小、耗时都不变', 9.4, RED[1])
    ax.plot([5.3, 5.3], [0.15, 5.3], color='#dee2e6', lw=1)

    label(ax, 11.5, 5.1, '结构化剪枝', 12, bold=True)
    dead = (1, 4)
    grid(6.3, 1.75, 6, 5, dead_rows=dead)
    label(ax, 7.35, 4.5, '第 1 层的权重', 9, '#868e96'); label(ax, 6.0, 3.01, '输\n出\n通\n道', 9, '#868e96')
    grid(9.4, 2.17, 4, 6, dead_cols=dead, color=GREEN)
    label(ax, 10.66, 4.5, '第 2 层的权重', 9, '#868e96')
    for r in dead:
        label(ax, 8.62, 1.75 + (5 - r) * c + c / 2, '×', 12, RED[1]); label(ax, 9.4 + r * c + c / 2, 4.05, '×', 12, RED[1])
    label(ax, 9.2, 1.15, '删掉第 1 层的输出通道，第 2 层对应的输入通道也一起删掉', 9.4)
    arrow(ax, (12.15, 3.01), (13.0, 3.01), lw=1.3)
    grid(13.2, 2.17, 4, 5); grid(15.65, 2.17, 4, 4, color=GREEN)
    label(ax, 15.3, 4.5, '两个更小的稠密矩阵', 9, '#868e96')
    label(ax, 11.5, 0.45, '重新搭一个小模型并复制保留的权重：参数量、文件大小、耗时都真正减少', 9.4, GREEN[1])
    save(fig, T, 'pruning.png')

# ---- 2. sparsity：准确率与稀疏度、剪枝前后的权重分布
if 'sparsity' in which:
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axs[0]; sp = D['sparsity']; xs = range(len(sp))
    ax.axhline(D['large_acc'], color='#868e96', ls='--', lw=1.2, label=f"剪枝前 {D['large_acc']:.3f}")
    ax.plot(xs, [s[1] for s in sp], color=RED[1], marker='x', ms=6, lw=1.8, label='剪枝后直接评估')
    ax.plot(xs, [s[2] for s in sp], color=BLUE[1], marker='o', ms=5, lw=1.8, label='再微调 10 轮')
    ax.set_xticks(list(xs)); ax.set_xticklabels([f'{s[0]:.0%}' for s in sp]); ax.set_xlabel('全局稀疏度'); ax.set_ylabel('测试集准确率')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(loc='lower left', frameon=False); ax.set_title('准确率与稀疏度', fontsize=11.5); tidy(ax)
    ax = axs[1]; e = np.array(D['hist_edges']); w = e[1] - e[0]
    ax.bar(e[:-1], D['hist_before'], width=w, align='edge', color='#ced4da', label='剪枝前')
    ax.bar(e[:-1], D['hist_after'], width=w, align='edge', color=BLUE[1], alpha=0.85, label='剪掉 90% 并微调后剩下的权重')
    for t in (-D['threshold'], D['threshold']): ax.axvline(t, color=RED[1], ls='--', lw=1.1)
    ax.text(D['threshold'] + 0.006, 2.2e4, f"阈值 ±{D['threshold']:.3f}", color=RED[1], fontsize=9.5)
    ax.set_yscale('log'); ax.set_xlim(-0.25, 0.25); ax.set_xlabel('权重的值'); ax.set_ylabel('个数（对数坐标）'); ax.legend(loc='upper left', frameon=False, fontsize=9.5)
    ax.set_ylim(0.8, 2e5); ax.set_title('四层权重合在一起的分布', fontsize=11.5); tidy(ax)
    fig.tight_layout(); save(fig, T, 'sparsity.png')

# ---- 3. distillation：软标签与多个随机种子的对比
if 'distillation' in which:
    fig = plt.figure(figsize=(11.6, 4.2)); gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 2.2, 2.4])
    s = D['sample']; ax = fig.add_subplot(gs[0])
    ax.imshow(s['image'], cmap='gray_r', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"标签：{s['label']}", fontsize=11)
    ax = fig.add_subplot(gs[1]); x = np.arange(10)
    ax.bar(x - 0.2, s['p1'], 0.4, color='#adb5bd', label='T = 1'); ax.bar(x + 0.2, s['p4'], 0.4, color=ORANGE[1], label='T = 4')
    ax.set_xticks(x); ax.set_xlabel('类别'); ax.set_ylabel('教师给出的概率'); ax.legend(frameon=False); ax.set_title('教师对这张图像的软标签', fontsize=11.5); tidy(ax)
    ax = fig.add_subplot(gs[2]); d = D['distill']
    groups = [('单独训练', d['alone'], GREY[1]), ('蒸馏', d['kd'], BLUE[1]), ('蒸馏\n加上无标注数据', d['kd_more'], GREEN[1])]
    for i, (name, v, c) in enumerate(groups):
        ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, color=c, s=34, alpha=0.75, zorder=3)
        m = np.mean(v); ax.plot([i - 0.25, i + 0.25], [m, m], color=c, lw=2.4, zorder=4); ax.text(i + 0.29, m, f'{m:.4f}', va='center', fontsize=9.5, color=c)
    ax.axhline(D['large_acc'], color=RED[1], ls='--', lw=1.2); ax.text(2.62, D['large_acc'] + 0.0012, '教师', color=RED[1], fontsize=9.5, ha='right')
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups]); ax.set_xlim(-0.5, 2.7); ax.set_ylim(0.938, 0.98)
    ax.set_ylabel('测试集准确率'); ax.grid(axis='y', alpha=0.3); ax.set_title('学生在 5 个随机种子下的结果，横线为平均值', fontsize=11.5); tidy(ax)
    fig.tight_layout(); save(fig, T, 'distillation.png')

# ---- 4. fake_quant：伪量化的前向与直通估计的梯度
if 'fake_quant' in which:
    scale, qmin, qmax = 0.25, -2, 2
    x = np.linspace(-1, 1, 2001); q = np.clip(np.round(x / scale), qmin, qmax) * scale
    inside = (np.round(x / scale) >= qmin) & (np.round(x / scale) <= qmax)
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax = axs[0]
    ax.plot(x, x, color='#adb5bd', ls='--', lw=1.3, label='不量化：y = x'); ax.plot(x, q, color=BLUE[1], lw=2, label='伪量化的输出')
    ax.axvspan(-1, -0.625, color=RED[0], alpha=0.7, lw=0); ax.axvspan(0.625, 1, color=RED[0], alpha=0.7, lw=0)
    ax.text(-0.81, 0.05, '超出范围\n被截断', ha='center', fontsize=9.5, color=RED[1]); ax.text(0.81, -0.25, '超出范围\n被截断', ha='center', fontsize=9.5, color=RED[1])
    ax.set_xlabel('输入 x'); ax.set_ylabel('输出'); ax.set_title('前向：5 个取值的阶梯（scale = 0.25）', fontsize=11.5); ax.legend(loc='upper left', frameon=False, fontsize=9.5); ax.grid(alpha=0.3); tidy(ax)
    ax = axs[1]
    ax.plot(x, inside.astype(float), color=ORANGE[1], lw=2, label='直通估计：范围内当作 1，范围外为 0')
    ax.plot(x, np.zeros_like(x), color=BLUE[1], lw=2, label='阶梯函数真正的梯度：几乎处处为 0')
    ax.set_ylim(-0.15, 1.45); ax.set_yticks([0, 1]); ax.set_xlabel('输入 x'); ax.set_ylabel('输出对输入的梯度'); ax.set_title('反向：用直通估计代替真正的梯度', fontsize=11.5)
    ax.legend(loc='upper center', frameon=False, fontsize=9.5); ax.grid(alpha=0.3); tidy(ax)
    fig.tight_layout(); save(fig, T, 'fake_quant.png')

# ---- 5. bits：训练后量化与量化感知训练在不同位宽下的准确率
if 'bits' in which:
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); b = D['bits']; xs = range(len(b))
    ax.axhline(np.mean(D['float_acc']), color='#868e96', ls='--', lw=1.2, label=f"浮点模型 {np.mean(D['float_acc']):.3f}")
    for k, name, c, m in [(1, '训练后量化', RED[1], 'x'), (2, '量化感知训练', BLUE[1], 'o')]:
        mean = [np.mean(r[k]) for r in b]
        ax.fill_between(xs, [min(r[k]) for r in b], [max(r[k]) for r in b], color=c, alpha=0.13, lw=0)
        ax.plot(xs, mean, color=c, marker=m, ms=6, lw=1.8, label=name)
        for i, v in enumerate(mean):
            if b[i][0] in (4, 3): ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(16, 7) if k == 2 else (-20, -13), ha='center', fontsize=9.5, color=c)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f'{r[0]} 位' for r in b]); ax.set_xlabel('权重和中间结果的位宽'); ax.set_ylabel('测试集准确率')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(loc='lower left', frameon=False); tidy(ax)
    ax.set_title('5 个学生模型的平均值，色带为最低到最高', fontsize=11.5)
    fig.tight_layout(); save(fig, T, 'bits.png')
