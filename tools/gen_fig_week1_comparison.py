"""Regenerate the Week 1 four-algorithm comparison figure — 1行5列 layout.
Reads the first defect overlay from each algorithm + original image.
Output: report/final_assets/fig_week1_four_algorithms_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import numpy as np
import os

ROOT = r"E:\leinaozuoye"
OUT = os.path.join(ROOT, "report", "final_assets", "fig_week1_four_algorithms_comparison.png")

# CJK font
for f in fm.fontManager.ttflist:
    if "Microsoft YaHei" in f.name:
        plt.rcParams["font.family"] = f.name
        break
plt.rcParams["axes.unicode_minus"] = False

# Load original and overlay images (use the first defect frame 000_color_000)
original_path = os.path.join(ROOT, "data", "mvtec", "carpet", "test", "color", "000.png")
original = Image.open(original_path).convert("RGB")

overlays = {}
algo_names = {"padim_carpet": "PaDiM", "patchcore_carpet": "PatchCore",
              "spade_carpet": "SPADE", "anomalydino_carpet": "AnomalyDINO"}
for algo_dir, label in algo_names.items():
    overlay_path = os.path.join(ROOT, "runs", algo_dir, "overlays", "0000_color_000_overlay.png")
    overlays[label] = Image.open(overlay_path).convert("RGB")

# Create 1×5 layout
fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
fig.patch.set_facecolor("#fafafa")

titles = ["Original (carpet 000)", "PaDiM", "PatchCore", "SPADE", "AnomalyDINO"]

for ax, title in zip(axes, titles):
    ax.set_facecolor("#f0f0f0")
    ax.set_xticks([])
    ax.set_yticks([])

# Original
axes[0].imshow(original)
axes[0].set_title(titles[0], fontsize=13, fontweight="bold", pad=8)

# Four overlays
for i, (algo_label, img) in enumerate(overlays.items()):
    axes[i + 1].imshow(img)
    axes[i + 1].set_title(titles[i + 1], fontsize=13, fontweight="bold", pad=8)

# Add sub-labels with memory bank type
sub_labels = [
    "(a) 原始缺陷图像",
    "(b) 统计式记忆\nMahalanobis距离",
    "(c) 代表性patch记忆\n最近邻距离",
    "(d) 样本式记忆\nk-NN检索+局部差分",
    "(e) 强特征token记忆\nDINOv2 pretrained NN",
]
for ax, sub in zip(axes, sub_labels):
    ax.set_xlabel(sub, fontsize=9, color="#444", labelpad=6)

fig.suptitle("第一周四类 Memory Bank 异常热力图对比 — MVTec AD carpet（缺陷图像 #000）",
             fontsize=15, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {OUT}")
print(f"Size: {os.path.getsize(OUT)} bytes")
