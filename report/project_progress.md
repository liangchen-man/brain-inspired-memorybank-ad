# 项目进度记录

更新时间：2026-06-14

## 当前阶段

项目已经完成数据、环境、PaDiM 教学版 carpet/bottle 实验、PatchCore 教学版 carpet/bottle 实验、SPADE 教学版 carpet 实验、AnomalyDINO 教学版 carpet 实验。当前阶段进入最终整理与验收。

## 已完成

- 数据集：`data/mvtec/carpet/` 已解压并通过结构检查。
- 数据集：`data/mvtec/bottle/` 已解压并通过结构检查。
- 环境：`braincv-ad-py310` 已创建，PyTorch CUDA 可用，GPU 为 RTX 3050 4GB Laptop。
- 脚本：`src/check_dataset.py` 已存在。
- 脚本：`src/check_env.py` 已存在。
- 脚本：`src/run_padim.py` 已存在。
- 脚本：`src/run_patchcore.py` 已存在。
- 脚本：`src/run_spade.py` 已存在。
- 脚本：`src/run_anomalydino.py` 已存在。
- 脚本：`src/collect_results.py` 已存在，可生成 `results/summary.csv`。
- 脚本：`src/make_figures_index.py` 已存在，可生成 `report/figures_index.md`。
- PaDiM + carpet：已完成，输出在 `runs/padim_carpet/`。
- PaDiM + bottle：已完成，输出在 `runs/padim_bottle/`。
- PatchCore-teaching + carpet：已完成，输出在 `runs/patchcore_carpet/`。
- PatchCore-teaching + bottle：已完成，输出在 `runs/patchcore_bottle/`。
- SPADE-teaching + carpet：已完成，输出在 `runs/spade_carpet/`。
- AnomalyDINO-teaching + carpet：已完成，输出在 `runs/anomalydino_carpet/`。
- 汇总表：`results/summary.csv` 已生成并更新。
- 图像索引：`report/figures_index.md` 已生成并更新。
- 分析草稿：`report/analysis_draft.md` 已创建。
- 最终报告：`report/final_report.md` 已创建。

## 当前真实实验结果

| Algorithm | Category | Status | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Heatmaps |
|---|---|---|---:|---:|---:|---:|---:|
| PaDiM-teaching | carpet | success | 0.99358 | 0.98616 | 231.634s | 31.936 MB | 30 |
| PaDiM-teaching | bottle | success | 0.99603 | 0.98004 | 69.473s | 31.936 MB | 30 |
| PatchCore-teaching | carpet | success | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 30 |
| PatchCore-teaching | bottle | success | 1.00000 | 0.97592 | 68.712s | 31.936 MB | 30 |
| SPADE-teaching | carpet | success | 0.97713 | 0.98196 | 209.005s | 31.936 MB | 30 |
| AnomalyDINO-teaching | carpet | success | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 30 |

## 风险与约束

- 当前 PaDiM、PatchCore、SPADE、AnomalyDINO 均为教学版或最小复现路线，报告中必须如实说明，不得写成完整官方 SOTA 复现。
- `external/` 仍为空，所有可运行结果来自 `src/` 下教学版脚本。
- Faiss 未安装；当前 PatchCore 教学版不依赖 Faiss。
- 所有实验输出已放在 `runs/` 下。
- 所有汇总结果已放在 `results/` 下。
- 报告素材已放在 `report/` 下。

## 下一步

1. 最终检查 `results/summary.csv`、`report/figures_index.md`、`report/final_report.md`。
2. 如果需要提交 Word/PDF，再把 `report/final_report.md` 转写成课程要求格式。
