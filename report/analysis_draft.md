# 实验分析草稿

题目：类脑计算对计算机视觉的启发：从工业异常检测中的 memory bank 范式理解正常性记忆

更新时间：2026-06-14

## 1. 当前实验状态

本项目已经完成 PaDiM 和 PatchCore 教学版在 MVTec AD `carpet` 与 `bottle` 两类上的实验，也完成了 SPADE 教学版和 AnomalyDINO 教学版在 `carpet` 上的最小实验。所有指标均来自 `runs/*/results.json`，没有编造未运行结果。

| Algorithm | Category | Status | Image AUROC | Pixel AUROC | Runtime | GPU Memory | Memory Bank Bytes |
|---|---|---|---:|---:|---:|---:|---:|
| PaDiM-teaching | carpet | success | 0.99358 | 0.98616 | 231.634s | 31.936 MB | 13049029 |
| PaDiM-teaching | bottle | success | 0.99603 | 0.98004 | 69.473s | 31.936 MB | 13049029 |
| PatchCore-teaching | carpet | success | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515205 |
| PatchCore-teaching | bottle | success | 1.00000 | 0.97592 | 68.712s | 31.936 MB | 515205 |
| SPADE-teaching | carpet | success | 0.97713 | 0.98196 | 209.005s | 31.936 MB | 56272133 |
| AnomalyDINO-teaching | carpet | success | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1029317 |

## 2. 正常性记忆思想

工业异常检测不一定要先学习缺陷。缺陷样本常常稀缺、长尾且不可穷尽，而正常样本更容易采集且模式更稳定。因此可以先建立正常样本的特征记忆，再把测试样本与正常记忆比较。若测试样本无法被正常记忆解释，就被视为异常。

这一点与类脑计算中的记忆启发相通：系统先形成对常态模式的记忆，再对偏离常态的新输入产生异常响应。memory bank 就是这种正常性记忆在计算机视觉中的工程化表达。

## 3. 各算法的 memory bank 解释

PaDiM 的记忆是统计记忆。它在每个空间位置保存正常特征的均值和逆协方差，用 Mahalanobis 距离判断测试特征是否偏离该位置的正常分布。

PatchCore 的记忆是显式 patch 记忆。它保留正常训练图像的 patch 特征，并用 coreset 压缩。测试 patch 到正常 patch memory bank 的最近邻距离越大，越可能异常。

SPADE 的记忆是实例近邻记忆。它先检索与测试图像相似的正常图像，再比较局部空间特征差异，因此 memory bank 中保存了正常图像的全局特征和空间特征图。

AnomalyDINO 的记忆是基于 DINOv2/ViT patch token 的正常 patch token 记忆。它使用自监督 DINOv2 预训练视觉表征，将正常图像切分为 patch tokens，保存正常 token memory bank，再用最近邻距离生成异常分数。

## 4. 结果分析

PaDiM 在 `carpet` 和 `bottle` 上都取得较高 AUROC，说明空间统计记忆对结构稳定的工业图像有效。PatchCore 的 memory bank 文件最小，因为它只保存 coreset 后的 2000 个 patch。SPADE 的 memory bank 最大，因为它保存每张正常图像的空间特征图。AnomalyDINO 使用 ViT-S DINOv2 backbone，显存峰值上升到 101.105 MB，但仍远低于 RTX 3050 4GB 上限，并且在 `carpet` 上取得当前最高的 image AUROC 和 pixel AUROC。

当前实验不应被解释为官方 SOTA 排名，因为 PaDiM、PatchCore、SPADE、AnomalyDINO 均为教学版或最小路线。它们的价值主要在于帮助理解不同形式的正常性记忆。

## 5. 3050 显卡评估

已完成实验的运行时间均低于 4 分钟，显存峰值最高为 AnomalyDINO 的 101.105 MB。当前课程实验规模下，RTX 3050 4GB Laptop 足够使用，不需要租 3090 或 4090。

## 6. 工业迁移限制

- 正常样本必须覆盖真实生产中的正常波动。
- 光照、相机、角度、背景和生产批次变化会导致分布漂移。
- 预训练特征可能与工业图像域不完全匹配。
- 阈值选择会影响误报和漏报，部署时需要结合业务成本。
- 教学版实现没有覆盖官方算法全部细节，不能直接代表工业级性能。
