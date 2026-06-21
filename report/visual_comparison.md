# 四算法同一样本热力图对比

本文件用于第一周报告的可视化增强，展示同一个 `carpet/color/000.png` 样本在四类 memory bank 方法下的定位结果。

## 对比图

![carpet color 000 algorithm comparison](E:/leinaozuoye/report/figures/carpet_color_000_algorithm_comparison.png)

## 文件路径

| Item | Path |
|---|---|
| Original | `data/mvtec/carpet/test/color/000.png` |
| GT mask | `data/mvtec/carpet/ground_truth/color/000_mask.png` |
| PaDiM overlay | `runs/padim_carpet/overlays/0000_color_000_overlay.png` |
| PatchCore overlay | `runs/patchcore_carpet/overlays/0000_color_000_overlay.png` |
| SPADE overlay | `runs/spade_carpet/overlays/0000_color_000_overlay.png` |
| AnomalyDINO overlay | `runs/anomalydino_carpet/overlays/0000_color_000_overlay.png` |

## 观察

- 四个方法都能在缺陷主体区域产生高响应，说明基于正常性记忆的异常定位在该样本上有效。
- PaDiM 的响应更依赖空间位置统计，热区较集中。
- PatchCore 使用 patch 最近邻距离，能较好覆盖缺陷边缘。
- SPADE 因为参考相似正常样本的空间特征图，响应范围更大。
- AnomalyDINO 使用 DINOv2 patch token，主缺陷区域响应清晰，说明强预训练特征对该纹理异常有帮助。

## 报告中可用的结论

> 同一样本的横向可视化表明，不同 memory bank 形式虽然都能定位缺陷，但热力图形态不同。PaDiM 更像位置统计偏离，PatchCore 和 AnomalyDINO 更像局部 patch/token 与正常记忆的距离，而 SPADE 更强调与相似正常样本的局部差异。这说明 memory bank 不只是一个存储结构，也决定了模型如何解释“偏离正常”。
