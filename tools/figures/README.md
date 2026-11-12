# 文章配图的生成脚本

`assets/img/posts/` 下由脚本生成的图片都可以用这里的脚本重新生成，重新生成的结果与现有文件逐字节相同。

```bash
python3 tools/figures/pytorch-cnn.py           # 重新生成一篇文章的图片
FIG_OUT=/tmp/figs/ python3 tools/figures/pytorch-cnn.py   # 输出到别处，不覆盖现有图片
```

- `lib.py`：公共的字体设置、配色和画框图用的函数（`canvas`、`box`、`arrow`、`label`、`save`）
- `data/<文章>/`：图中的曲线和样本所用的数据，来自文章中代码的实际运行结果。数据变了要先重新导出，再重新画图
- `misc-diagrams.py`、`misc-plots.py`：早期多篇文章的示意图和曲线图
- `image-processing.py`：图像滤波、形态学、边缘检测的效果对比图
- 其余脚本与文章同名

需要 `matplotlib`、`numpy`，`image-processing.py` 还需要 `opencv-python`；中文字体使用 Noto Sans CJK。

截图类的图片（JetBrains、OpenOCD、`nvidia-smi` 等）不是脚本生成的。
