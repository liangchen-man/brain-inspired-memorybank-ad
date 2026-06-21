# PatchCore 教学版实验汇总

本项目当前使用的是 `src/run_patchcore.py` 中的教学版 PatchCore，而不是官方 `amazon-science/patchcore-inspection` 的完整复现。该版本用于课程实验中的 memory bank 解释：先用正常样本提取 patch 特征，构建正常 patch 的记忆库，再用测试 patch 到记忆库最近邻的距离生成异常分数和热力图。

## 实验设置

| Item | Value |
|---|---|
| Backbone | ResNet18 layer2 |
| Feature dim | 64 |
| Coreset method | random |
| Coreset patches | 2000 |
| NN distance | `torch.cdist` |
| Faiss | Not used |
| Device | CUDA, RTX 3050 4GB Laptop |

## 结果

| Category | Status | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Train images | Raw patches | Coreset patches | Memory bank bytes | Heatmaps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet | success | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 280 | 219520 | 2000 | 515205 | 30 |
| bottle | success | 1.00000 | 0.97592 | 68.712s | 31.936 MB | 209 | 163856 | 2000 | 515205 | 30 |

## 报告中应如实说明

- 该实现保留了 PatchCore 的核心思想：patch-level memory bank、coreset 压缩、nearest-neighbor anomaly score。
- 该实现不是官方完整复现，未使用 wide ResNet、Faiss、官方采样和完整评测脚本。
- carpet 比 bottle 更耗时，主要因为训练图像和 patch 数量更多。
- 当前 RTX 3050 运行时间可接受，无需租用 3090。
