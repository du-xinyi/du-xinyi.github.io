import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import *
import numpy as np

# ---- G. transform frames
fig, ax = plt.subplots(figsize=(6.4, 5.6))
def frame(o, th, name, c, L=4):
    o = np.array(o); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for k, lab in enumerate(['x', 'y']):
        e = o + R[:, k] * L
        ax.annotate('', e, o, arrowprops=dict(arrowstyle='-|>', color=c, lw=2))
        ax.text(*(e + R[:, k] * 0.5), f'${lab}_{name}$', color=c, fontsize=12, ha='center', va='center')
    ax.plot(*o, 'o', color=c, ms=5)
frame((0, 0), 0, 'A', '#2f6fd0'); frame((12, 6), np.deg2rad(30), 'B', '#d9822b')
th = np.deg2rad(30); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
o = np.array([12, 6]); p = o + R @ np.array([3, 7])
ax.plot(*p, 'o', color='#d64545', ms=7, zorder=5); ax.text(p[0] + 0.4, p[1] + 0.3, r'$\mathbf{p}$', color='#d64545', fontsize=13)
ax.annotate('', o, (0, 0), arrowprops=dict(arrowstyle='-|>', color='#868e96', lw=1.3, ls='--')); ax.text(5.2, 3.6, r'${}^{A}\mathbf{p}_{B_o}=(12,\ 6)$', color='#495057', fontsize=10.5, rotation=26.5, ha='center')
ax.annotate('', p, o, arrowprops=dict(arrowstyle='-|>', color='#d9822b', lw=1.3, ls='--')); ax.text(12.4, 10.6, r'${}^{B}\mathbf{p}=(3,\ 7)$', color='#d9822b', fontsize=10.5)
ax.annotate('', p, (0, 0), arrowprops=dict(arrowstyle='-|>', color='#d64545', lw=1.3, ls='--')); ax.text(3.2, 6.6, r'${}^{A}\mathbf{p}=(11.098,\ 13.562)$', color='#d64545', fontsize=10.5, rotation=50.5, ha='center')
arc = np.linspace(0, th, 30); ax.plot(12 + 2.2 * np.cos(arc), 6 + 2.2 * np.sin(arc), color='#d9822b', lw=1.2); ax.plot([12, 15.5], [6, 6], color='#adb5bd', lw=1, ls=':')
ax.text(14.6, 6.55, r'$30^{\circ}$', color='#d9822b', fontsize=10.5)
ax.set_xlim(-1.5, 18); ax.set_ylim(-1.5, 15.5); ax.set_aspect('equal'); ax.grid(True, ls=':', color='#dee2e6')
ax.set_xticks(range(0, 19, 3)); ax.set_yticks(range(0, 16, 3)); fig.tight_layout()
save(fig, '2024-07-05-homogeneous-transformation-matrix', 'frames.png')

# ---- H. fov
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
f, W = 3.0, 4.4
a1.plot([-f, -f], [-W / 2, W / 2], color='#2f6fd0', lw=5, solid_capstyle='butt'); a1.text(-f - 0.25, 0, '传感器', color='#2f6fd0', rotation=90, va='center', ha='right', fontsize=11)
a1.plot(0, 0, 'o', color=INK, ms=6); a1.text(0.05, -0.45, '镜头光心', fontsize=10, ha='center')
for sgn in (1, -1):
    a1.plot([-f, 4.2], [-sgn * W / 2, sgn * W / 2 * 4.2 / f], color='#d9822b', lw=1.4)
a1.plot([-f, 4.4], [0, 0], color='#adb5bd', lw=1, ls='-.')
ang = np.linspace(-np.arctan(W / 2 / f), np.arctan(W / 2 / f), 40); a1.plot(1.5 * np.cos(ang), 1.5 * np.sin(ang), color='#d64545', lw=1.5); a1.text(1.75, 0.12, r'$\theta$', color='#d64545', fontsize=14)
a1.annotate('', (-f, -W / 2 - 0.45), (0, -W / 2 - 0.45), arrowprops=dict(arrowstyle='<->', color='#495057')); a1.text(-f / 2, -W / 2 - 0.85, r'焦距 $f$', ha='center', fontsize=11)
a1.annotate('', (-f - 0.95, W / 2), (-f - 0.95, -W / 2), arrowprops=dict(arrowstyle='<->', color='#495057')); a1.text(-f - 1.15, 0.9, r'$W$', ha='right', fontsize=12)
a1.text(3.0, 0.75, '视场范围', color='#d9822b', fontsize=10.5)
a1.text(0.6, -3.35, r'$\tan\frac{\theta}{2}=\frac{W/2}{f}$', fontsize=13)
a1.set_xlim(-5.2, 4.8); a1.set_ylim(-3.9, 3.6); a1.set_aspect('equal'); a1.axis('off'); a1.set_title('标准镜头的视场角', fontsize=12)
al = np.linspace(0, np.pi / 2, 300)
a2.plot(np.degrees(al[:260]), np.tan(al[:260]), label=r'针孔模型 $r=f\tan\alpha$', color='#868e96', ls='--')
a2.plot(np.degrees(al), 2 * np.tan(al / 2), label=r'体视投影 $r=2f\tan\frac{\alpha}{2}$', color='#7a4fd0')
a2.plot(np.degrees(al), al, label=r'等距投影 $r=f\alpha$', color='#2f6fd0')
a2.plot(np.degrees(al), 2 * np.sin(al / 2), label=r'等立体角投影 $r=2f\sin\frac{\alpha}{2}$', color='#2e9950')
a2.plot(np.degrees(al), np.sin(al), label=r'正交投影 $r=f\sin\alpha$', color='#d9822b')
a2.set_ylim(0, 3); a2.set_xlim(0, 90); a2.set_xticks(range(0, 91, 15)); a2.grid(True, ls=':')
a2.set_xlabel(r'入射角 $\alpha$（度）'); a2.set_ylabel(r'像高 $r\,/\,f$'); a2.legend(fontsize=9.5, loc='upper left'); a2.set_title('不同投影模型下像高随入射角的变化', fontsize=12)
fig.tight_layout(); save(fig, '2024-10-05-field-of-view', 'fov.png')

# ---- I. matplotlib
M = '2025-01-06-matplotlib-visualization'
x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), 'b-', label='sin(x)'); ax.plot(x, np.cos(x), color='red', linestyle='--', label='cos(x)')
ax.set_title('Trigonometric Functions'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(True, linestyle=':'); ax.legend()
os.makedirs(ROOT + M, exist_ok=True)
fig.savefig(ROOT + M + '/trig.png', dpi=130, bbox_inches='tight'); plt.close(fig); print('trig')

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
for i, m in enumerate(markers):
    r, c = divmod(i, 5)
    a1.plot(c, -r, marker=m, ms=13, color='#2f6fd0', mec='#1c4a94', ls='')
    a1.text(c + 0.22, -r, f"'{m}'", va='center', fontsize=11, family='Noto Sans Mono CJK SC')
a1.set_xlim(-0.4, 4.9); a1.set_ylim(-4.6, 0.6); a1.axis('off'); a1.set_title('标记 marker', fontsize=12)
for i, (ls, name) in enumerate([('-', "'-'  solid"), ('--', "'--'  dashed"), ('-.', "'-.'  dashdot"), (':', "':'  dotted")]):
    a2.plot([0, 1], [-i, -i], ls=ls, color='#2f6fd0', lw=2.2); a2.text(0, -i + 0.2, name, fontsize=11, family='Noto Sans Mono CJK SC')
a2.set_xlim(-0.05, 1.05); a2.set_ylim(-3.5, 0.7); a2.axis('off'); a2.set_title('线型 linestyle', fontsize=12)
hs = ['/', '\\', '-', '|', '+', 'x', 'o', '.', '*']
for i, h in enumerate(hs):
    r, c = divmod(i, 3)
    a3.add_patch(Rectangle((c * 1.25, -r * 1.25), 1, 0.8, hatch=h * 2, fc='#e7f0fd', ec='#2f6fd0'))
    a3.text(c * 1.25 + 0.5, -r * 1.25 - 0.2, f"'{h}'", ha='center', fontsize=11, family='Noto Sans Mono CJK SC')
a3.set_xlim(-0.1, 3.6); a3.set_ylim(-3.0, 1.0); a3.set_aspect('equal'); a3.axis('off'); a3.set_title('填充图案 hatch', fontsize=12)
fig.tight_layout(); save(fig, M, 'styles.png')

cm = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds', 'YlOrRd', 'YlGnBu', 'RdBu', 'RdYlGn', 'Spectral', 'Set1', 'Set2', 'Paired']
fig, axs = plt.subplots(len(cm), 1, figsize=(7.5, 6.2)); g = np.linspace(0, 1, 256)[None, :]
for a, n in zip(axs, cm):
    a.imshow(g, aspect='auto', cmap=n); a.axis('off'); a.text(-0.015, 0.5, n, transform=a.transAxes, ha='right', va='center', fontsize=10.5)
fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.02, hspace=0.35); save(fig, M, 'colormaps.png')

# ---- J. ansi colors
P = ['#000000', '#cd0000', '#00cd00', '#cdcd00', '#0000ee', '#cd00cd', '#00cdcd', '#e5e5e5']
B = ['#7f7f7f', '#ff0000', '#00ff00', '#ffff00', '#5c5cff', '#ff00ff', '#00ffff', '#ffffff']
names = ['黑', '红', '绿', '黄', '蓝', '洋红', '青', '白']
fig, ax = canvas(12, 5.2, 0.85); ax.add_patch(Rectangle((0, 0), 12, 5.2, fc='#1e1e1e'))
mono = dict(family='Noto Sans Mono CJK SC', va='center')
rows = [(4.5, '前景色 30~37', P, 30, 'fg'), (3.6, '亮前景色 90~97', B, 90, 'fg'), (2.7, '背景色 40~47', P, 40, 'bg'), (1.8, '亮背景色 100~107', B, 100, 'bg')]
for y, t, pal, base, kind in rows:
    ax.text(0.3, y, t, color='#d4d4d4', fontsize=10.5, **mono)
    for i, c in enumerate(pal):
        x = 3.0 + i * 1.1
        if kind == 'bg':
            ax.add_patch(Rectangle((x - 0.05, y - 0.27), 1.0, 0.54, fc=c))
            ax.text(x + 0.45, y, f'{base + i}', color='#000000' if i not in (0, 4) or base == 100 else '#ffffff', fontsize=11, ha='center', **mono)
        else:
            if c == '#000000': ax.add_patch(Rectangle((x - 0.05, y - 0.27), 1.0, 0.54, fc='#4a4a4a'))
            ax.text(x + 0.45, y, f'{base + i}', color=c, fontsize=11, ha='center', fontweight='bold', **mono)
for i, n in enumerate(names): ax.text(3.45 + i * 1.1, 1.2, n, color='#9a9a9a', fontsize=9.5, ha='center', va='center')
ax.text(0.3, 0.5, '样式', color='#d4d4d4', fontsize=10.5, **mono)
ax.text(3.0, 0.5, '1 粗体', color='#e5e5e5', fontsize=11, fontweight='bold', **mono)
ax.text(5.0, 0.5, '4 下划线', color='#e5e5e5', fontsize=11, **mono); ax.plot([5.0, 6.28], [0.3, 0.3], color='#e5e5e5', lw=1)
ax.add_patch(Rectangle((7.15, 0.23), 1.3, 0.54, fc='#e5e5e5')); ax.text(7.25, 0.5, '7 反显', color='#1e1e1e', fontsize=11, **mono)
ax.text(9.2, 0.5, '1;31 粗体红色', color='#ff0000', fontsize=11, fontweight='bold', **mono)
save(fig, '2024-05-30-formatted-output', 'ansi_colors.png')

# ---- L. conv
fig, ax = canvas(12, 5.4, 0.85)
x0, y0, s = 0.5, 0.7, 0.55
for r in range(7):
    for c in range(7):
        pad = r in (0, 6) or c in (0, 6)
        ax.add_patch(Rectangle((x0 + c * s, y0 + (6 - r) * s), s, s, fc='#f1f3f5' if pad else '#e7f0fd', ec='#adb5bd' if pad else '#2f6fd0', lw=0.9))
        if pad: ax.text(x0 + c * s + s / 2, y0 + (6 - r) * s + s / 2, '0', ha='center', va='center', fontsize=8.5, color='#adb5bd')
def win(r, c, col): ax.add_patch(Rectangle((x0 + c * s, y0 + (6 - r - 2) * s), 3 * s, 3 * s, fill=False, ec=col, lw=2.6, zorder=5))
win(0, 0, '#d64545'); win(0, 2, '#d9822b'); win(2, 0, '#2e9950')
label(ax, x0 + 3.5 * s, y0 + 7 * s + 0.35, '输入 5×5，padding = 1', 11, bold=True); label(ax, x0 + 3.5 * s, 0.3, '灰色为填充的 0，彩色框为 3×3 卷积核的位置', 9.2, '#868e96')
ox, oy, t = 7.2, 1.75, 0.8
for r in range(3):
    for c in range(3):
        col = {(0, 0): RED, (0, 1): ORANGE, (1, 0): GREEN}.get((r, c), ('#ffffff', '#868e96'))
        ax.add_patch(Rectangle((ox + c * t, oy + (2 - r) * t), t, t, fc=col[0], ec=col[1], lw=1.6 if (r, c) in [(0, 0), (0, 1), (1, 0)] else 1))
label(ax, ox + 1.5 * t, oy + 3 * t + 0.35, '输出 3×3', 11, bold=True)
arrow(ax, (x0 + 7 * s + 0.25, 2.95), (ox - 0.25, 2.95), 'kernel_size = 3\nstride = 2', off=(0, 0.45), fs=10, lw=1.8)
arrow(ax, (x0 + 3 * s, y0 + 7 * s - 0.1), (x0 + 5 * s, y0 + 7 * s - 0.1), None, color='#d9822b', rad=-0.5, lw=1.3)
ax.text(ox + 3 * t + 0.35, 2.95, r'$\left\lfloor\frac{5 + 2\times1 - 3}{2}\right\rfloor + 1 = 3$', fontsize=13, va='center')
save(fig, '2024-07-10-pytorch-cnn', 'conv_output.png')
