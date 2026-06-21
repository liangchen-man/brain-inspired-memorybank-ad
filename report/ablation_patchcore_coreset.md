# PatchCore Coreset Size 消融实验

本消融用于回答一个部署相关问题：如果减少 PatchCore memory bank 中的代表性 patch 数量，性能是否会明显下降？

## 实验设置

两组实验均使用 `carpet` 类别和相同的 ResNet18 layer2 特征。唯一区别是 coreset size：

| Item | Value |
|---|---|
| Dataset | MVTec AD carpet |
| Backbone | ResNet18 layer2 |
| Feature dim | 64 |
| Coreset method | random |
| Raw patch count | 219520 |
| Candidate patch count | 20000 |
| Heatmaps | 30 |
| Overlays | 30 |

## 结果

| Coreset size | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory bank bytes |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515205 |
| 1000 | 0.95104 | 0.98130 | 180.701s | 31.936 MB | 259205 |

## 观察

- Coreset 从 2000 减到 1000 后，memory bank 文件从 515205 bytes 减到 259205 bytes，约减半。
- Image AUROC 从 0.95947 降到 0.95104，下降约 0.008，属于轻微下降。
- Pixel AUROC 从 0.98151 降到 0.98130，几乎无变化。
- Runtime 从 215.411s 降到 180.701s，约快 16%。

## 结论

该消融说明：在当前 `carpet` 实验中，PatchCore 的 random coreset 即使减半，仍能保留大部分正常性表示能力。对于真实工位异常监控，这意味着 memory bank 可以通过 coreset 控制规模，从而降低存储和检索成本；但 coreset 过小也可能损失少数正常模式，因此第二周迁移时应结合现场正常样本多样性选择 coreset 大小。

报告中可以写成：

> PatchCore 的 coreset 并不是简单压缩文件大小，而是在部署成本和正常模式覆盖之间做权衡。本次消融中，coreset 减半几乎不影响像素级定位指标，说明代表性 patch 记忆有利于真实工业场景的轻量部署。
