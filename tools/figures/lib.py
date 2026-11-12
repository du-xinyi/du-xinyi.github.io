import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
# 图片输出到 assets/img/posts/<文章目录>/；设置环境变量 FIG_OUT 可以改到别处，便于试画
ROOT = os.environ.get('FIG_OUT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'img', 'posts') + '/'
BLUE = ('#e7f0fd', '#2f6fd0'); GREEN = ('#e6f6ea', '#2e9950'); ORANGE = ('#fdf0e1', '#d9822b')
GREY = ('#f1f3f5', '#868e96'); RED = ('#fdeaea', '#d64545'); PURPLE = ('#f1eafd', '#7a4fd0')
INK = '#343a40'

def canvas(w, h, scale=1.0):
    fig, ax = plt.subplots(figsize=(w * scale, h * scale))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect('equal'); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax

def box(ax, x, y, w, h, text='', c=BLUE, fs=11, bold=False, mono=False, lw=1.4, round=0.08, tc=INK, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={round}', fc=c[0], ec=c[1], lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc,
                fontweight='bold' if bold else 'normal', family='Noto Sans Mono CJK SC' if mono else None, linespacing=1.35)

def arrow(ax, p1, p2, text=None, color='#495057', rad=0.0, fs=9.5, off=(0, 0.13), ls='-', lw=1.4, style='-|>', tc=None, ha='center'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], text, ha=ha, va='center', fontsize=fs, color=tc or color)

def label(ax, x, y, text, fs=10, color=INK, ha='center', va='center', bold=False, mono=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, fontweight='bold' if bold else 'normal',
            family='Noto Sans Mono CJK SC' if mono else None, linespacing=1.35)

def save(fig, post, name, dpi=130):
    d = ROOT + post; os.makedirs(d, exist_ok=True)
    fig.savefig(f'{d}/{name}', dpi=dpi, facecolor='white'); plt.close(fig)
    print(post, name, os.path.getsize(f'{d}/{name}') // 1024, 'KB')
