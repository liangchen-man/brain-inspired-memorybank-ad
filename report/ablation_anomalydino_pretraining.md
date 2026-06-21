# AnomalyDINO 预训练特征消融实验

本消融用于回答一个关键问题：AnomalyDINO 的优势来自 ViT 结构本身，还是来自 DINOv2 自监督预训练特征？

## 实验设置

两组实验使用完全相同的结构和 memory bank 参数：

| Item | Value |
|---|---|
| Dataset | MVTec AD carpet |
| Model | `vit_small_patch14_dinov2` |
| Image size | 224 |
| Token grid | 16 x 16 |
| Feature dim | 128 |
| Coreset size | 2000 |
| Raw patch count | 71680 |
| Heatmaps | 30 |
| Overlays | 30 |

唯一区别是是否加载 DINOv2 pretrained 权重。

## 结果

| Setting | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory bank bytes |
|---|---:|---:|---:|---:|---:|
| pretrained=True | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1029317 |
| pretrained=False | 0.33427 | 0.67811 | 203.251s | 101.105 MB | 1029317 |

## 观察

- 去掉预训练后，Image AUROC 从 0.99559 降到 0.33427，下降约 0.661。
- Pixel AUROC 从 0.98767 降到 0.67811，下降约 0.310。
- GPU 显存和 memory bank 文件大小保持一致，说明差异不是由模型结构或 memory bank 规模造成的。
- non-pretrained 版本耗时更长，但没有带来有效特征表达。

## 结论

该消融说明：AnomalyDINO 的有效性主要来自 DINOv2 自监督预训练特征，而不是单纯来自 ViT 结构或 memory bank 机制。对于第二周真实工位异常监控，这一点非常重要：如果要迁移到复杂现场，强预训练特征能显著提升正常性记忆的可用性。

报告中可以写成：

> 在相同 ViT 架构、相同 coreset 大小和相同 memory bank 文件大小下，未加载 DINOv2 预训练权重的版本几乎失去图像级异常检测能力。这说明“强特征记忆”是 AnomalyDINO 相比普通 patch memory 的核心价值，也提示真实工业迁移中应优先使用具备泛化能力的预训练视觉表征。
