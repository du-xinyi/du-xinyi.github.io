import os

HERE = os.path.dirname(os.path.abspath(__file__))
# 不导入 lib.py：文章中的示例使用 matplotlib 的默认字体
ROOT = os.environ.get('FIG_OUT') or os.path.join(HERE, '..', '..', 'assets', 'img', 'posts') + '/'

# 与文章“绘制多图”一节的示例代码相同，只是把图片保存到博客的图片目录
M = '2025-01-06-matplotlib-visualization'
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
x = np.linspace(0, 2 * np.pi, 50)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

axes[0, 0].plot(x, np.sin(x), 'b-', label='sin(x)')
axes[0, 0].plot(x, np.cos(x), 'r--', label='cos(x)')
axes[0, 0].set_title('plot')
axes[0, 0].legend()

points = rng.normal(size=(100, 2))
axes[0, 1].scatter(points[:, 0], points[:, 1], c=points[:, 0], cmap='viridis', s=20)
axes[0, 1].set_title('scatter')

axes[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 5, 2], color='C1', edgecolor='black')
axes[1, 0].set_title('bar')

axes[1, 1].hist(rng.normal(size=1000), bins=30, color='C2', alpha=0.8)
axes[1, 1].set_title('hist')

fig.tight_layout()
os.makedirs(ROOT + M, exist_ok=True)
fig.savefig(ROOT + M + '/subplots.png', dpi=150)
