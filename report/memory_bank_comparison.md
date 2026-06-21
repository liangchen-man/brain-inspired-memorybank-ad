# Memory Bank 对比与迁移能力分析

本文件用于把第一周实验从“复现实验结果”提升为“第二周工位异常监控迁移的技术基线”。

## 四类记忆形式

| Algorithm | Memory bank 形式 | 保存内容 | 异常判断方式 | 适合的工业场景 | 迁移到工位监控的风险 |
|---|---|---|---|---|---|
| PaDiM | 统计式记忆 | 每个空间位置的均值与逆协方差 | Mahalanobis 距离 | 固定机位、固定结构、位置稳定 | 工件位置变化或相机偏移会导致误报 |
| SPADE | 样本式记忆 | 正常图像全局特征与空间特征图 | 先检索相似正常图像，再比较局部差异 | 正常状态少、需要相似参考样本解释 | memory bank 大，检索成本较高 |
| PatchCore | 代表性 patch 记忆 | 正常 patch 特征的 coreset | 测试 patch 到正常 patch 最近邻距离 | 局部缺陷、工具/工件局部异常 | 对流程顺序和时序异常不敏感 |
| AnomalyDINO | 强特征 patch token 记忆 | DINOv2 patch token coreset | 测试 token 到正常 token 最近邻距离 | 复杂场景、外观变化较多、需要泛化特征 | 依赖预训练特征，解释性弱于 ResNet 层级特征 |

## 实验结果与部署成本

| Algorithm | Category | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory bank bytes |
|---|---|---:|---:|---:|---:|---:|
| PaDiM-teaching | carpet | 0.99358 | 0.98616 | 231.634s | 31.936 MB | 13049029 |
| PatchCore-teaching | carpet | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515205 |
| SPADE-teaching | carpet | 0.97713 | 0.98196 | 209.005s | 31.936 MB | 56272133 |
| AnomalyDINO-teaching | carpet | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1029317 |

## 对第二周工位异常监控的指导

- 如果工位机位固定、产品位置稳定，PaDiM 是最容易解释的基线。
- 如果工位异常表现为局部缺失、遮挡、工具摆放错误，PatchCore 是更稳的轻量 baseline。
- 如果工位背景复杂、工件外观变化较大，AnomalyDINO 更值得优先尝试。
- 如果报告需要说明“找相似正常状态”，SPADE 最直观。
- 第二周不必四个算法全跑，建议优先 PatchCore + AnomalyDINO，再用 PaDiM 作为固定位置对照。

## AnomalyDINO 预训练消融对迁移的启发

| Setting | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory bank bytes |
|---|---:|---:|---:|---:|---:|
| pretrained=True | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1029317 |
| pretrained=False | 0.33427 | 0.67811 | 203.251s | 101.105 MB | 1029317 |

该消融说明：同样的 ViT 结构和同样大小的 memory bank，并不自动带来好的异常检测能力。真正关键的是 DINOv2 自监督预训练特征。对于第二周工位异常监控，如果现场外观复杂、正常状态变化较多，应该优先使用 pretrained DINOv2，而不是随机初始化的 ViT。

## PatchCore coreset 消融对部署的启发

| Coreset size | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory bank bytes |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515205 |
| 1000 | 0.95104 | 0.98130 | 180.701s | 31.936 MB | 259205 |

该消融说明：PatchCore 的 coreset size 从 2000 减到 1000 后，memory bank 近似减半，运行时间下降，而 pixel AUROC 几乎不变。这对真实工位部署很重要：当正常样本越来越多时，可以通过 coreset 控制正常性记忆的规模，避免 memory bank 无限膨胀。

## 顶尖报告中的表达

第一周的结论不应写成“哪个算法最高”，而应写成：

> 四类算法分别体现了统计式记忆、样本式记忆、代表性 patch 记忆和强特征 token 记忆。它们共同说明工业异常检测可以先建立正常性记忆，再判断测试样本是否偏离正常模式。第二周工位异常监控的核心问题，是这些静态正常性记忆能否适应真实现场的光照、遮挡、位置变化和时序状态变化。
