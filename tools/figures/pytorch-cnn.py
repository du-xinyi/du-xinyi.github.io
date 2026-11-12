import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import *
fig, ax = canvas(18.2, 8.2, 0.72)
def flow(x, title, items, w=2.6, top=7.0, gap=0.28, h=0.62):
    label(ax, x + w / 2, top + 0.75, title, 11.5, bold=True)
    ys = []
    y = top
    for text, c in items:
        box(ax, x, y - h, w, h, text, c, fs=9.6); ys.append((y, y - h)); y -= h + gap
    for (a, b), (c2, d) in zip(ys, ys[1:]): arrow(ax, (x + w / 2, b), (x + w / 2, c2), lw=1.2)
    return ys
# 1 ConvBNReLU
ys = flow(0.4, 'Conv-BN-ReLU', [('输入', GREY), ('Conv2d 3×3\nbias=False', BLUE), ('BatchNorm2d', GREEN), ('ReLU', ORANGE), ('输出', GREY)])
# 2 BasicBlock
x = 4.0
ys = flow(x, '残差块 BasicBlock', [('输入', GREY), ('Conv2d 3×3', BLUE), ('BN + ReLU', GREEN), ('Conv2d 3×3', BLUE), ('BN', GREEN), ('相加', PURPLE), ('ReLU', ORANGE), ('输出', GREY)])
sx = x + 2.6 + 0.75
ax.plot([x + 2.6, sx, sx], [ys[0][0] - 0.31, ys[0][0] - 0.31, ys[5][0] - 0.31], color='#7a4fd0', lw=1.4)
arrow(ax, (sx, ys[5][0] - 0.31), (x + 2.6, ys[5][0] - 0.31), color='#7a4fd0')
label(ax, sx + 0.12, (ys[0][0] + ys[5][1]) / 2, '捷径\n形状变化时\n用 1×1 卷积', 8.6, '#7a4fd0', ha='left')
# 3 Bottleneck
x = 8.6
ys = flow(x, '瓶颈块 Bottleneck', [('输入', GREY), ('Conv2d 1×1 降维\nBN + ReLU', BLUE), ('Conv2d 3×3\nBN + ReLU', BLUE), ('Conv2d 1×1 升维\nBN', BLUE), ('相加', PURPLE), ('ReLU', ORANGE), ('输出', GREY)], h=0.72)
sx = x + 2.6 + 0.6
ax.plot([x + 2.6, sx, sx], [ys[0][0] - 0.36, ys[0][0] - 0.36, ys[4][0] - 0.36], color='#7a4fd0', lw=1.4)
arrow(ax, (sx, ys[4][0] - 0.36), (x + 2.6, ys[4][0] - 0.36), color='#7a4fd0')
label(ax, sx + 0.12, (ys[0][0] + ys[4][1]) / 2, '捷径', 8.6, '#7a4fd0', ha='left')
# 4 SE
x = 13.9
ys = flow(x, '通道注意力 SE', [('输入', GREY), ('全局平均池化', GREEN), ('Linear 降维\nReLU', BLUE), ('Linear 升维\nSigmoid', BLUE), ('逐通道相乘', PURPLE), ('输出', GREY)], h=0.72)
sx = x - 0.7
ax.plot([x, sx, sx], [ys[0][0] - 0.36, ys[0][0] - 0.36, ys[4][0] - 0.36], color='#7a4fd0', lw=1.4)
arrow(ax, (sx, ys[4][0] - 0.36), (x, ys[4][0] - 0.36), color='#7a4fd0')
label(ax, x + 2.75, (ys[3][0] + ys[3][1]) / 2, '每个通道\n一个权重', 8.6, '#868e96', ha='left')
label(ax, sx - 0.12, (ys[0][0] + ys[4][1]) / 2, '原特征图', 8.6, '#7a4fd0', ha='right')
save(fig, '2024-07-10-pytorch-cnn', 'blocks.png')

# receptive field illustration: two stacked 3x3 convs see 5x5
fig, ax = canvas(11.5, 4.3, 0.85)
def grid(x0, y0, n, s, fill, title):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * s, y0 + r * s), s, s, fc=fill(r, c), ec='#adb5bd', lw=0.9))
    label(ax, x0 + n * s / 2, y0 + n * s + 0.3, title, 10.5, bold=True)
s = 0.5
grid(0.5, 0.6, 5, s, lambda r, c: '#e7f0fd', '输入 5×5')
grid(5.0, 1.1, 3, s, lambda r, c: '#e6f6ea', '第一层输出 3×3')
grid(9.2, 1.6, 1, s, lambda r, c: '#fdeaea', '第二层输出 1×1')
ax.add_patch(Rectangle((0.5, 1.6), 3 * s, 3 * s, fill=False, ec='#d9822b', lw=2.4))
ax.add_patch(Rectangle((5.0, 2.1), s, s, fill=False, ec='#d9822b', lw=2.4))
arrow(ax, (2.0, 2.6), (5.0, 2.35), '3×3 卷积', color='#d9822b', off=(0, 0.3), fs=9.5)
ax.add_patch(Rectangle((5.0, 1.1), 3 * s, 3 * s, fill=False, ec='#d64545', lw=2.0, ls='--'))
arrow(ax, (6.5, 1.85), (9.2, 1.85), '3×3 卷积', color='#d64545', off=(0, 0.25), fs=9.5)
label(ax, 5.75, 0.3, '第二层输出的一个点对应第一层的 3×3，再对应输入的 5×5，感受野为 5', 9.5, '#868e96')
save(fig, '2024-07-10-pytorch-cnn', 'receptive_field.png')
