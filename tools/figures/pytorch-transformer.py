import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import *
import numpy as np
T = '2026-11-11-pytorch-transformer'

def flow(ax, x, items, w=2.8, top=7.0, gap=0.3, h=0.62, fs=9.8):
    ys = []; y = top
    for text, c in items:
        box(ax, x, y - h, w, h, text, c, fs=fs); ys.append((y, y - h)); y -= h + gap
    for (a, b), (c2, d) in zip(ys, ys[1:]): arrow(ax, (x + w / 2, b), (x + w / 2, c2), lw=1.2)
    return ys

# ---- 1. attention
fig, ax = canvas(15, 7.6, 0.78)
label(ax, 3.3, 7.25, '缩放点积注意力', 12, bold=True)
box(ax, 0.6, 6.0, 1.5, 0.6, 'Q\n(N, L, E)', BLUE, fs=8.8); box(ax, 2.55, 6.0, 1.5, 0.6, 'K\n(N, S, E)', BLUE, fs=8.8); box(ax, 4.9, 6.0, 1.5, 0.6, 'V\n(N, S, E)', GREEN, fs=8.8)
ys = flow(ax, 1.2, [('矩阵相乘  $QK^{T}$', ORANGE), ('除以 $\\sqrt{E}$', ORANGE), ('掩码（可选）', GREY), ('softmax', ORANGE)], w=2.3, top=5.3, h=0.56, gap=0.32)
arrow(ax, (1.35, 6.0), (2.0, 5.3), lw=1.2); arrow(ax, (3.3, 6.0), (2.7, 5.3), lw=1.2)
label(ax, 3.75, 5.02, '(N, L, S)', 8.6, '#868e96', ha='left'); label(ax, 3.75, 2.38, '注意力权重\n(N, L, S)', 8.6, '#868e96', ha='left')
box(ax, 2.6, 0.9, 2.3, 0.56, '矩阵相乘', ORANGE, fs=9.8)
arrow(ax, (2.35, ys[3][1]), (3.3, 1.46), lw=1.2); arrow(ax, (5.65, 6.0), (4.3, 1.46), lw=1.2)
box(ax, 2.6, 0.05, 2.3, 0.5, '输出 (N, L, E)', PURPLE, fs=9.2); arrow(ax, (3.75, 0.9), (3.75, 0.55), lw=1.2)
ax.plot([7.0, 7.0], [0.1, 7.3], color='#dee2e6', lw=1)
label(ax, 11.0, 7.25, '多头注意力', 12, bold=True)
box(ax, 9.5, 6.0, 3.0, 0.6, 'Q、K、V  (N, L, E)', BLUE, fs=9.2)
box(ax, 9.5, 5.0, 3.0, 0.6, '各自经过一个线性层', GREY, fs=9.2); arrow(ax, (11.0, 6.0), (11.0, 5.6), lw=1.2)
for i, x in enumerate([7.6, 9.4, 12.6]):
    box(ax, x, 3.2, 1.6, 0.9, f'第 {["1", "2", "H"][i]} 个头\n(N, L, E/H)', ORANGE, fs=8.8)
    arrow(ax, (11.0, 5.0), (x + 0.8, 4.1), lw=1.1); arrow(ax, (x + 0.8, 3.2), (11.0, 2.3), lw=1.1)
label(ax, 11.8, 3.65, '…', 14)
label(ax, 14.35, 3.65, '每个头独立做\n缩放点积注意力', 8.6, '#868e96', ha='left')
box(ax, 9.5, 1.7, 3.0, 0.6, '拼接  (N, L, E)', GREY, fs=9.2)
box(ax, 9.5, 0.75, 3.0, 0.6, '线性层', GREY, fs=9.2); arrow(ax, (11.0, 1.7), (11.0, 1.35), lw=1.2)
box(ax, 9.5, 0.05, 3.0, 0.45, '输出 (N, L, E)', PURPLE, fs=9.2); arrow(ax, (11.0, 0.75), (11.0, 0.5), lw=1.2)
ax.set_xlim(0, 16.4); fig.set_size_inches(16.4 * 0.78, 7.6 * 0.78)
save(fig, T, 'attention.png')

# ---- 2. masks
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8))
words = ['我', '喜欢', '机器', '学习', '<pad>']
def grid(ax, allowed, title, xl, yl):
    n = len(words)
    for r in range(n):
        for c in range(n):
            ok = allowed(r, c)
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fc='#e6f6ea' if ok else '#f1f3f5', ec='white', lw=2))
            ax.text(c + 0.5, n - 1 - r + 0.5, '✓' if ok else '×', ha='center', va='center', fontsize=13, color='#2e9950' if ok else '#adb5bd')
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect('equal')
    ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels(words); ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels(words[::-1])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top'); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel(xl, labelpad=8); ax.set_ylabel(yl, labelpad=8); ax.set_title(title, y=-0.13, fontsize=11.5)
grid(axs[0], lambda r, c: c <= r, '因果掩码 attn_mask：只能看到自己和之前的位置', '被关注的位置（Key）', '当前位置（Query）')
grid(axs[1], lambda r, c: c < 4, '填充掩码 key_padding_mask：所有位置都不看填充', '被关注的位置（Key）', '当前位置（Query）')
fig.tight_layout(w_pad=3); save(fig, T, 'masks.png')

# ---- 3. positional encoding
d, L = 128, 100
pos = np.arange(L)[:, None]; div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
pe = np.zeros((L, d)); pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div)
fig, ax = plt.subplots(figsize=(9, 4.2))
im = ax.imshow(pe, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
ax.set_xlabel('编码的维度'); ax.set_ylabel('位置'); ax.set_title('正弦位置编码：每一行是一个位置的编码向量', fontsize=11.5)
fig.colorbar(im, ax=ax, pad=0.02); fig.tight_layout(); save(fig, T, 'positional_encoding.png')

# ---- 4. layers
fig, ax = canvas(17, 9.2, 0.72)
def stack(x, title, items, w=3.0, top=8.0, h=0.62, gap=0.3):
    label(ax, x + w / 2, top + 0.7, title, 11.5, bold=True)
    return flow(ax, x, items, w=w, top=top, h=h, gap=gap, fs=9.4)
def skip(x, w, ys, a, b, side=1, off=0.5):
    sx = x + w + off if side > 0 else x - off
    ex = x + w if side > 0 else x
    ya = (ys[a][0] + ys[a][1]) / 2 - 0.13; yb = (ys[b][0] + ys[b][1]) / 2 + 0.13
    ax.plot([ex, sx, sx], [ya, ya, yb], color='#7a4fd0', lw=1.3); arrow(ax, (sx, yb), (ex, yb), color='#7a4fd0', lw=1.3)
ATT = BLUE; FF = GREEN; NORM = ORANGE; ADD = PURPLE
x = 0.5
ys = stack(x, '编码器层（后归一化）', [('输入', GREY), ('多头自注意力', ATT), ('相加', ADD), ('LayerNorm', NORM), ('前馈网络', FF), ('相加', ADD), ('LayerNorm', NORM), ('输出', GREY)])
skip(x, 3.0, ys, 0, 2); skip(x, 3.0, ys, 3, 5, off=0.85)
x = 5.3
ys = stack(x, '编码器层（前归一化）', [('输入', GREY), ('LayerNorm', NORM), ('多头自注意力', ATT), ('相加', ADD), ('LayerNorm', NORM), ('前馈网络', FF), ('相加', ADD), ('输出', GREY)])
skip(x, 3.0, ys, 0, 3); skip(x, 3.0, ys, 3, 6, off=0.85)
x = 12.0
ys = stack(x, '解码器层（前归一化）', [('输入（目标序列）', GREY), ('LayerNorm', NORM), ('带因果掩码的自注意力', ATT), ('相加', ADD), ('LayerNorm', NORM), ('交叉注意力', ATT), ('相加', ADD), ('LayerNorm', NORM), ('前馈网络', FF), ('相加', ADD), ('输出', GREY)], h=0.56, gap=0.2, top=8.0)
skip(x, 3.0, ys, 0, 3); skip(x, 3.0, ys, 3, 6, off=0.85); skip(x, 3.0, ys, 6, 9)
box(ax, 9.9, (ys[5][0] + ys[5][1]) / 2 - 0.45, 1.5, 0.9, '编码器\n的输出', GREY, fs=8.8)
arrow(ax, (11.4, (ys[5][0] + ys[5][1]) / 2), (12.0, (ys[5][0] + ys[5][1]) / 2), color='#2f6fd0', lw=1.3)
label(ax, 10.65, (ys[5][0] + ys[5][1]) / 2 - 0.75, '作为 K 和 V', 8.4, '#2f6fd0')
label(ax, 4.3, 0.2, '紫色连线为残差连接：子层的输入直接加到子层的输出上', 9.2, '#868e96')
ax.set_ylim(-0.6, 9.2); fig.set_size_inches(17 * 0.72, 9.8 * 0.72)
save(fig, T, 'layers.png')


# ---- shift
fig, ax = canvas(11.5, 3.9, 0.85)
ins = ['我', '喜欢', '机器', '学习']; outs = ['喜欢', '机器', '学习', '<结束>']
label(ax, 1.0, 3.05, '目标', 10.5, bold=True); label(ax, 1.0, 0.75, '输入', 10.5, bold=True)
box(ax, 2.2, 1.55, 8.6, 0.7, '带因果掩码的 Transformer', GREY, fs=10.5)
for i, (a, b) in enumerate(zip(ins, outs)):
    x = 2.4 + i * 2.15
    box(ax, x, 0.4, 1.75, 0.7, a, BLUE, fs=10.5); box(ax, x, 2.7, 1.75, 0.7, b, GREEN, fs=10.5)
    arrow(ax, (x + 0.875, 1.1), (x + 0.875, 1.55), lw=1.2); arrow(ax, (x + 0.875, 2.25), (x + 0.875, 2.7), lw=1.2)
label(ax, 5.75, 3.7, '每个位置的输出用来预测它的下一个词，一次前向传播同时训练所有位置', 9.4, '#868e96')
save(fig, T, 'shift.png')

# ---- kv cache
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9))
words = ['我', '喜欢', '机器', '学习', '很']
def grid(ax, active, title):
    n = len(words)
    for r in range(n):
        for c in range(n):
            if c > r: fc, t, col = '#ffffff', '', '#ffffff'
            elif active(r): fc, t, col = '#fdf0e1', '计算', '#d9822b'
            else: fc, t, col = '#f1f3f5', '跳过', '#adb5bd'
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fc=fc, ec='white' if c <= r else '#f1f3f5', lw=2))
            ax.text(c + 0.5, n - 1 - r + 0.5, t, ha='center', va='center', fontsize=10, color=col)
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect('equal')
    ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels(words); ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels(words[::-1])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top'); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel('Key / Value', labelpad=8); ax.set_ylabel('Query', labelpad=8); ax.set_title(title, y=-0.14, fontsize=11.5)
grid(axs[0], lambda r: True, '不使用缓存：每生成一个词，所有位置重新计算')
grid(axs[1], lambda r: r == 4, '使用缓存：只计算新词这一行，K、V 取自缓存')
fig.tight_layout(w_pad=3); save(fig, T, 'kv_cache.png')

# ---- rope
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
theta = np.deg2rad(22); q0 = np.deg2rad(15); k0 = np.deg2rad(50)
def circ(ax, m, n, title):
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color='#dee2e6', lw=1)
    for ang, c, name, pos in [(q0 + m * theta, '#2f6fd0', 'q', m), (k0 + n * theta, '#d9822b', 'k', n)]:
        ax.annotate('', (np.cos(ang), np.sin(ang)), (0, 0), arrowprops=dict(arrowstyle='-|>', color=c, lw=2.2))
        ax.text(1.2 * np.cos(ang), 1.2 * np.sin(ang), f'{name}（位置 {pos}）', color=c, ha='center', va='center', fontsize=10.5)
    a = np.linspace(q0 + m * theta, k0 + n * theta, 40); ax.plot(0.42 * np.cos(a), 0.42 * np.sin(a), color='#d64545', lw=2)
    mid = (q0 + m * theta + k0 + n * theta) / 2; ax.text(0.6 * np.cos(mid), 0.6 * np.sin(mid), '夹角', color='#d64545', ha='center', va='center', fontsize=10)
    ax.plot([-1.25, 1.25], [0, 0], color='#ced4da', lw=0.8); ax.plot([0, 0], [-1.25, 1.25], color='#ced4da', lw=0.8)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.45); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=11, color='#495057')
circ(axs[0], 1, 3, 'q 在位置 1，k 在位置 3'); circ(axs[1], 5, 7, 'q 在位置 5，k 在位置 7')
fig.suptitle('每个向量按自己的位置旋转；相对位置相同（都相差 2）时夹角相同，点积也相同', fontsize=11.5, y=0.97)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, T, 'rope.png')
