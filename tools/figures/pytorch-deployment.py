import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys, json
from lib import *
import numpy as np
T = '2026-11-11-pytorch-deployment'
W = HERE + '/data/pytorch-deployment/'
which = sys.argv[1:] or ['pipeline', 'preprocess', 'quant', 'latency']

if 'pipeline' in which:
    fig, ax = canvas(16, 6.6, 0.78)
    for x, t in [(1.5, '训练'), (5.6, '导出'), (9.9, '模型文件'), (14.0, '推理')]:
        label(ax, x, 6.25, t, 11.5, bold=True)
    box(ax, 0.3, 2.6, 2.4, 1.2, '训练好的模型\nstate_dict (.pth)', BLUE, fs=9.6)
    rows = [(4.9, 'load_state_dict\neval()', None, 'Python + PyTorch\nCPU / GPU', GREEN),
            (3.5, 'torch.export.export', '.pt2', 'Python + PyTorch\nExecuTorch 等', GREEN),
            (2.1, 'torch.onnx.export', '.onnx', 'ONNX Runtime（可再量化为 int8）\nPython / C++，CPU / GPU', ORANGE),
            (0.4, None, None, None, None)]
    for y, exp, f, run, c in rows[:3]:
        box(ax, 4.1, y, 3.0, 0.9, exp, GREY, fs=9.2, mono=False)
        arrow(ax, (2.7, 3.2), (4.1, y + 0.45), lw=1.2)
        if f:
            box(ax, 8.9, y, 2.0, 0.9, f, c, fs=10)
            arrow(ax, (7.1, y + 0.45), (8.9, y + 0.45), lw=1.2); arrow(ax, (10.9, y + 0.45), (12.1, y + 0.45), lw=1.2)
        else:
            arrow(ax, (7.1, y + 0.45), (12.1, y + 0.45), lw=1.2)
        box(ax, 12.1, y, 3.7, 0.9, run, c, fs=9.2)
    # onnx branches
    box(ax, 8.9, 0.4, 2.0, 0.9, '编译引擎', GREY, fs=10); arrow(ax, (9.9, 2.1), (9.9, 1.3), lw=1.2)
    arrow(ax, (10.9, 0.85), (12.1, 0.85), lw=1.2)
    box(ax, 12.1, 0.4, 3.7, 0.9, 'TensorRT (.engine)\nNVIDIA GPU', PURPLE, fs=9.2)
    save(fig, T, 'pipeline.png')

if 'preprocess' in which:
    fig, ax = canvas(17.2, 5.0, 0.76)
    steps = [('读入图像', 'cv2.imread', '(H, W, 3)\nuint8，BGR', '用 PIL 读入\n则是 RGB'),
             ('缩放', '短边到 256', '(256, W\', 3)\nuint8', '插值方式\n与训练时不同'),
             ('中心裁剪', '224 × 224', '(224, 224, 3)\nuint8', '裁剪位置\n差一个像素'),
             ('通道顺序', 'BGR → RGB', '(224, 224, 3)\nuint8，RGB', '忘记转换'),
             ('缩放数值', '除以 255', '(224, 224, 3)\nfloat32，[0, 1]', '忘记除以 255\n或留在 uint8'),
             ('标准化', '减均值、除标准差', '(224, 224, 3)\nfloat32', '均值、标准差\n与训练时不同'),
             ('调整维度', 'HWC → NCHW', '(1, 3, 224, 224)\nfloat32', '忘记换维\n或内存不连续')]
    w, gap = 2.1, 0.33
    label(ax, 0.2, 4.72, '步骤', 9.5, '#868e96', ha='left'); label(ax, 0.2, 2.75, '输出', 9.5, '#868e96', ha='left'); label(ax, 0.2, 1.05, '常见错误', 9.5, '#d64545', ha='left')
    for i, (a, b, shape, bug) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, 3.3, w, 1.15, f'{a}\n{b}', BLUE, fs=9.6)
        box(ax, x, 1.6, w, 0.95, shape, GREY, fs=8.6)
        arrow(ax, (x + w / 2, 3.3), (x + w / 2, 2.55), lw=1.1, color='#adb5bd')
        box(ax, x, 0.05, w, 0.8, bug, RED, fs=8.6, tc='#b23a3a')
        if i < len(steps) - 1: arrow(ax, (x + w, 3.875), (x + w + gap, 3.875), lw=1.3)
    save(fig, T, 'preprocess.png')

if 'quant' in which:
    d = np.load(W + 'quant_fig.npz'); w = d['w']; err = d['err']
    lo, hi = w.min(), w.max(); scale = (hi - lo) / 255
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.9))
    ax = axs[0]
    ax.hist(w, bins=120, color='#2f6fd0', alpha=0.85)
    for v, t, ha in [(lo, f'最小值 {lo:.3f}\n对应整数 0', 'left'), (hi, f'最大值 {hi:.3f}\n对应整数 255', 'right')]:
        ax.axvline(v, color='#d64545', lw=1.2, ls='--'); ax.text(v + (0.012 if ha == 'left' else -0.012), ax.get_ylim()[1] * 0.80, t, color='#d64545', fontsize=9, ha=ha)
    ax.set_xlabel('权重的取值（float32）'); ax.set_ylabel('个数'); ax.set_title('ResNet-18 全连接层的 51.2 万个权重', fontsize=11)
    ax = axs[1]
    ax.hist(err / scale, bins=80, color='#d9822b', alpha=0.85)
    ax.set_xlabel('量化误差 ÷ 量化步长'); ax.set_ylabel('个数'); ax.set_title(f'量化到 256 个等间距的值，步长 {scale:.4f}', fontsize=11)
    ax.set_xlim(-0.6, 0.6)
    for a in axs:
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    fig.tight_layout(w_pad=2.5); save(fig, T, 'quantization.png')

if 'latency' in which:
    R = HERE + '/data/pytorch-deployment/'
    import re
    r = {}
    for l in open(R + 'out/s8_bench.out'):
        m = re.match(r'(.+?)\s+batch=(\d+)\s+([\d.]+) ms', l); r[m.group(1), int(m.group(2))] = float(m.group(3))
    for l in open(R + 'out/trt_test.out'):
        m = re.match(r'TensorRT batch=(\d+)\s+([\d.]+) ms', l)
        if m: r['TensorRT', int(m.group(1))] = float(m.group(2))
    warm = json.load(open(R + 'bench.json'))['warm']
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.1), gridspec_kw={'width_ratios': [1.25, 1, 1.1]})
    def bars(ax, names, short, cols, title):
        xs = np.arange(len(names)); bw = 0.38
        v1 = [r[n, 1] for n in names]; v8 = [r[n, 8] / 8 for n in names]
        ax.bar(xs - bw / 2, v1, bw, color=cols); ax.bar(xs + bw / 2, v8, bw, color=cols, alpha=0.45, hatch='//', edgecolor='white')
        for x, v in zip(xs - bw / 2, v1): ax.text(x, v + max(v1) * 0.015, f'{v:.2f}', ha='center', fontsize=8.4)
        for x, v in zip(xs + bw / 2, v8): ax.text(x, v + max(v1) * 0.015, f'{v:.2f}', ha='center', fontsize=8.4, color='#868e96')
        ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=8.8); ax.set_ylabel('每张图像的耗时（毫秒）'); ax.set_title(title, fontsize=10.5)
    bars(axs[0], ['PyTorch CPU', 'ONNX Runtime CPU', 'ONNX Runtime CPU 动态量化', 'ONNX Runtime CPU 静态量化'],
         ['PyTorch', 'ONNX Runtime', 'ONNX Runtime\n动态量化', 'ONNX Runtime\n静态量化'], ['#2f6fd0', '#d9822b', '#e0a868', '#b5651d'], 'CPU：实色为批大小 1，斜线为批大小 8')
    bars(axs[1], ['PyTorch GPU', 'ONNX Runtime GPU', 'TensorRT'], ['PyTorch', 'ONNX Runtime', 'TensorRT'], ['#2e9950', '#d9822b', '#7a4fd0'], 'GPU：实色为批大小 1，斜线为批大小 8')
    ax = axs[2]
    for n, c in [('PyTorch CPU', '#2f6fd0'), ('ONNX Runtime CPU', '#d9822b'), ('PyTorch GPU', '#2e9950')]:
        ax.plot(np.arange(1, len(warm[n]) + 1), warm[n], marker='o', ms=3.2, lw=1.4, color=c, label=n)
    ax.set_yscale('log'); ax.set_xlabel('第几次调用'); ax.set_ylabel('单次耗时（毫秒，对数坐标）'); ax.legend(frameon=False, fontsize=9)
    ax.set_title('加载模型后的前 30 次调用（批大小 1）', fontsize=10.5)
    for a in axs:
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    fig.tight_layout(w_pad=2.2); save(fig, T, 'latency.png')
