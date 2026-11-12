import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import *
import cv2
import numpy as np

# 图像滤波、形态学运算、边缘检测三张效果对比图，使用合成的测试图
rng = np.random.default_rng(0)
H, W = 300, 400


def scene():
    x = np.linspace(60, 150, W, dtype=np.float32)
    img = np.repeat(np.tile(x, (H, 1))[:, :, None], 3, axis=2).astype(np.uint8)
    cv2.rectangle(img, (30, 40), (150, 140), (60, 90, 220), -1)
    cv2.circle(img, (280, 95), 60, (70, 200, 230), -1)
    cv2.fillPoly(img, [np.array([[60, 270], [130, 170], [200, 270]])], (200, 150, 60))
    cv2.putText(img, 'OpenCV', (215, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 3, cv2.LINE_AA)
    for i in range(5):
        cv2.line(img, (230 + i * 30, 255), (245 + i * 30, 285), (20, 20, 20), 1, cv2.LINE_AA)
    return img


def show(axs, items):
    for ax, (t, im) in zip(axs.ravel(), items):
        if im.ndim == 2:
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=12)
        ax.axis('off')


def save_jpg(fig, post, name):
    """先存为png再转成jpg，带噪声的图用jpg体积小得多"""
    d = ROOT + post
    os.makedirs(d, exist_ok=True)
    tmp = f'{d}/{name}.tmp.png'
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    cv2.imwrite(f'{d}/{name}.jpg', cv2.imread(tmp), [cv2.IMWRITE_JPEG_QUALITY, 88])
    os.remove(tmp)
    print(post, name + '.jpg', os.path.getsize(f'{d}/{name}.jpg') // 1024, 'KB')


clean = scene()
noisy = clean.astype(np.float32) + rng.normal(0, 12, clean.shape)
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
m = rng.random((H, W))
noisy[m < 0.01] = 0
noisy[m > 0.99] = 255

# 滤波
items = [('原图', clean), ('加入高斯噪声与椒盐噪声', noisy),
         ('均值滤波 blur 5x5', cv2.blur(noisy, (5, 5))),
         ('高斯滤波 GaussianBlur 5x5', cv2.GaussianBlur(noisy, (5, 5), 0)),
         ('中值滤波 medianBlur 5', cv2.medianBlur(noisy, 5)),
         ('双边滤波 bilateralFilter 9, 75, 75', cv2.bilateralFilter(noisy, 9, 75, 75))]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
show(axs, items)
fig.tight_layout()
save_jpg(fig, '2023-12-09-image-filtering', 'filter_compare')

# 形态学
b = np.zeros((H, W), np.uint8)
cv2.rectangle(b, (30, 40), (150, 140), 255, -1)
cv2.circle(b, (280, 95), 60, 255, -1)
cv2.putText(b, 'OpenCV', (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 8, cv2.LINE_AA)
cv2.line(b, (150, 90), (220, 90), 255, 2)
pts = rng.integers(0, [W, H], (60, 2))
for x, y in pts[:30]:
    cv2.circle(b, (int(x), int(y)), 1, 255, -1)
for x, y in pts[30:]:
    if b[y, x]:
        cv2.circle(b, (int(x), int(y)), 2, 0, -1)
for x, y in [(70, 80), (110, 110), (260, 70), (300, 120), (285, 90)]:
    cv2.circle(b, (x, y), 3, 0, -1)
_, b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
items = [('原图', b), ('膨胀 dilate', cv2.dilate(b, k)), ('腐蚀 erode', cv2.erode(b, k)),
         ('开运算 MORPH_OPEN', cv2.morphologyEx(b, cv2.MORPH_OPEN, k)),
         ('闭运算 MORPH_CLOSE', cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)),
         ('形态学梯度 MORPH_GRADIENT', cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k))]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
show(axs, items)
fig.tight_layout()
save(fig, '2023-12-09-image-filtering', 'morphology_compare.png', dpi=110)

# 边缘检测，流程与文章中的示例代码相同
src = np.clip(clean.astype(np.float32) + rng.normal(0, 5, clean.shape), 0, 255).astype(np.uint8)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
bl = cv2.GaussianBlur(gray, (3, 3), 0)
gx = cv2.Sobel(bl, cv2.CV_16S, 1, 0)
gy = cv2.Sobel(bl, cv2.CV_16S, 0, 1)
sob = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
lap = cv2.convertScaleAbs(cv2.Laplacian(bl, cv2.CV_16S, ksize=3))
can = cv2.Canny(bl, 50, 150)
items = [('原图', src), ('Sobel', sob), ('Laplacian', lap), ('Canny 50, 150', can)]
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
show(axs, items)
fig.tight_layout()
save_jpg(fig, '2023-12-10-edge-detection', 'edge_compare')
