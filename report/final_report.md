# 类脑计算对计算机视觉的启发：基于 Memory Bank 范式的工业异常检测实验

## 摘要

本实验以 MVTec AD 数据集中的 `carpet` 和 `bottle` 为主要对象，围绕工业异常检测中的 memory bank 范式展开。实验目标不是追求完整 SOTA 复现，而是通过 PaDiM、PatchCore、SPADE 和 AnomalyDINO 四类方法理解“正常性记忆”：在缺陷样本稀缺、缺陷类型不可穷尽的工业场景中，模型可以先建立正常样本的特征记忆，再通过测试样本与正常记忆的偏离程度判断异常。

当前实验完成了 PaDiM-teaching、PatchCore-teaching 在 `carpet` 和 `bottle` 上的实验，以及 SPADE-teaching、AnomalyDINO-teaching 在 `carpet` 上的最小实验。所有结果均来自实际运行的 `runs/*/results.json`，未运行项目不编造指标。

## 1. 实验目的

工业缺陷检测与普通图像分类不同。普通分类通常依赖大量标注样本，而工业缺陷具有偶发性、长尾性和开放性：新的缺陷类型可能在部署后才出现，完整收集所有缺陷类别并不现实。因此，本实验从 memory bank 角度解释一种更适合工业场景的思路：先学习正常样本的结构、纹理和局部特征，再把偏离正常记忆的区域视为异常。

从类脑计算角度看，这种范式可以理解为一种“正常性记忆”。系统并不是先记住所有异常，而是形成对正常模式的内部表征。当新的输入无法被正常记忆很好解释时，系统给出异常分数和热力图。

## 2. 数据集与环境

数据集使用 MVTec AD。当前已解压并检查的类别如下：

| Category | Train/good | Test/good | Test defect | GT mask | Status |
|---|---:|---:|---:|---:|---|
| carpet | 280 | 28 | 89 | 89 | usable |
| bottle | 209 | 20 | 63 | 63 | usable |

环境使用 Anaconda + PyCharm 兼容路线。当前主要环境信息如下：

| Item | Value |
|---|---|
| Conda env | `braincv-ad-py310` |
| Python | 3.10.20 |
| PyTorch | 2.12.0+cu126 |
| torchvision | 0.27.0+cu126 |
| GPU | NVIDIA GeForce RTX 3050 4GB Laptop |
| CUDA available | Yes |
| OpenCV | 4.13.0 |
| scikit-learn | 1.7.2 |
| timm | 1.0.27 |
| Faiss | Not installed |

## 3. 方法设计

### 3.1 PaDiM：空间位置上的统计记忆

PaDiM 将正常训练图像映射到深度特征空间，并在每一个空间位置估计正常特征的高斯分布。测试时，若某个位置的特征相对于该位置的正常分布具有较大的 Mahalanobis 距离，则该位置更可能异常。

本项目教学版使用 ResNet18 layer2 特征，保存每个空间位置的均值和逆协方差。它体现的是压缩统计记忆：不保留全部训练 patch，而是把正常模式压缩为参数。

### 3.2 PatchCore：显式 patch 记忆库

PatchCore 将正常训练图像中的 patch 特征保存到 memory bank，再通过 coreset 压缩得到代表性正常 patch 集合。测试时，每个测试 patch 与正常 memory bank 计算最近邻距离，距离越大越异常。

本项目教学版使用 ResNet18 layer2 patch 特征、random coreset 和 `torch.cdist` 最近邻距离。它不是官方 PatchCore 完整复现，但保留了 patch memory bank、coreset 和 nearest-neighbor scoring 三个核心思想。

### 3.3 SPADE：相似正常样本检索与局部差异定位

SPADE 的直觉是先找与测试图像相似的正常参考图像，再比较局部特征差异。本项目教学版保存每张正常图像的全局特征和空间特征图。测试时先用全局特征检索相似正常样本，再用局部空间特征差异生成异常热力图。

这种方法体现的是实例级正常记忆：memory bank 中保留的是正常样本本身的特征表示，异常判断依赖“当前样本是否能被相似正常样本解释”。

### 3.4 AnomalyDINO：DINO patch token 记忆

AnomalyDINO 的核心启发是使用 DINOv2 等自监督视觉表征。相比普通监督分类特征，自监督特征可能更关注结构和局部一致性。本项目使用 `timm` 中的 `vit_small_patch14_dinov2` 预训练模型提取 patch tokens，构建正常 patch token memory bank，再用最近邻距离生成异常分数。

该实现不是官方 AnomalyDINO 仓库复现，而是教学版 AnomalyDINO-style 实验。它保留了 DINOv2 patch token、正常 token memory bank 和 nearest-neighbor scoring 的核心解释路径。

## 4. 实验结果

当前真实结果如下，来自 `runs/*/results.json` 和 `results/summary.csv`。

| Algorithm | Category | Status | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory Bank Bytes | Heatmaps |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PaDiM-teaching | carpet | success | 0.99358 | 0.98616 | 231.634s | 31.936 MB | 13049029 | 30 |
| PaDiM-teaching | bottle | success | 0.99603 | 0.98004 | 69.473s | 31.936 MB | 13049029 | 30 |
| PatchCore-teaching | carpet | success | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515205 | 30 |
| PatchCore-teaching | bottle | success | 1.00000 | 0.97592 | 68.712s | 31.936 MB | 515205 | 30 |
| SPADE-teaching | carpet | success | 0.97713 | 0.98196 | 209.005s | 31.936 MB | 56272133 | 30 |
| AnomalyDINO-teaching | carpet | success | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1029317 | 30 |

热力图和 overlay 索引见 `report/figures_index.md`。示例路径：

| Algorithm | Category | Example overlay |
|---|---|---|
| PaDiM-teaching | carpet | `runs/padim_carpet/overlays/0000_color_000_overlay.png` |
| PatchCore-teaching | carpet | `runs/patchcore_carpet/overlays/0000_color_000_overlay.png` |
| SPADE-teaching | carpet | `runs/spade_carpet/overlays/0000_color_000_overlay.png` |
| AnomalyDINO-teaching | carpet | `runs/anomalydino_carpet/overlays/0000_color_000_overlay.png` |

## 5. 结果分析

PaDiM 在 `carpet` 和 `bottle` 上都取得较高的 image AUROC 和 pixel AUROC，说明空间位置统计记忆对于结构稳定的工业样本有效。它的 memory bank 文件约 13 MB，保存的是每个空间位置的统计参数。

PatchCore 的 memory bank 文件最小，约 515 KB，因为它使用了 2000 个 coreset patch。其 `carpet` image AUROC 低于 PaDiM，但 pixel AUROC 仍较高，说明 patch 最近邻距离对定位缺陷区域仍然有效。

SPADE 的 memory bank 最大，约 56.27 MB，因为它保存了每张正常图像的全局特征和空间特征图。它在 `carpet` 上的 pixel AUROC 为 0.98196，与 PaDiM/PatchCore 接近，支持“相似正常样本检索 + 局部差异定位”的解释。

AnomalyDINO-teaching 使用 ViT-S DINOv2 预训练特征，在 `carpet` 上取得当前最高的 image AUROC 和 pixel AUROC。它的显存峰值为 101.105 MB，高于 ResNet18 教学版方法，但仍远低于 RTX 3050 4GB 的容量。其 memory bank 文件约 1.03 MB，因为最终保存的是 coreset 后的 DINO patch token。

为了验证 AnomalyDINO 的优势是否来自 DINOv2 预训练特征，本项目额外进行了 pretrained vs non-pretrained 消融。在相同 ViT 结构、相同 token grid、相同 coreset 大小和相同 memory bank 文件大小下，pretrained=True 的 image AUROC 为 0.99559，而 pretrained=False 只有 0.33427；pixel AUROC 也从 0.98767 降到 0.67811。这说明 AnomalyDINO 的关键价值不是“用了 ViT”本身，而是 DINOv2 自监督预训练提供了更适合迁移的视觉表征。

为了分析 memory bank 规模对部署的影响，本项目还进行了 PatchCore coreset size 消融。将 coreset 从 2000 减到 1000 后，memory bank 文件从 515205 bytes 降到 259205 bytes，runtime 从 215.411s 降到 180.701s；与此同时，pixel AUROC 仅从 0.98151 降到 0.98130，几乎不变。这说明代表性 patch 记忆可以在较小存储成本下保留主要正常模式，对真实工位场景的轻量部署具有启发意义。

当前实验不应被解释为官方 SOTA 排名，因为四类算法均为教学版或最小路线。它们的价值主要在于帮助理解不同形式的正常性记忆。

## 6. 真实工业迁移限制

Memory bank 范式适合缺陷样本少、正常样本多的场景，但真实工业部署仍面临限制：

- 正常样本必须覆盖生产中的正常波动。
- 光照、相机、角度和批次变化可能造成分布漂移。
- 阈值需要结合业务误报和漏报成本重新选择。
- 预训练特征不一定完全适合工业图像域。
- 当前实现是教学版，不代表官方完整复现或工业级性能。

## 7. 结论

本实验说明，工业异常检测不一定要先学习缺陷。通过 memory bank 建立正常样本的特征记忆，再判断测试样本是否偏离正常模式，可以在缺陷样本稀缺时提供可解释的异常检测路线。PaDiM、PatchCore、SPADE 和 AnomalyDINO 分别体现了统计记忆、patch 显式记忆、实例近邻记忆和 DINO patch token 记忆。

当前课程实验规模下，RTX 3050 4GB Laptop 足够完成本项目实验，不需要租用 3090 或 4090。

## 8. 面向第二周的迁移准备

为了将第一周实验自然过渡到第二周工位异常监控，本项目进一步整理了四个附录：

- `report/memory_bank_comparison.md`：四类 memory bank 的迁移能力对比。
- `report/transfer_to_workstation.md`：从 MVTec AD 到工位异常监控的迁移方案。
- `report/visual_comparison.md`：同一 carpet 样本的四算法热力图对比。
- `report/qa_appendix.md`：第一周任务清单逐题回答。
- `report/ablation_anomalydino_pretraining.md`：DINOv2 预训练特征消融实验。
- `report/ablation_patchcore_coreset.md`：PatchCore coreset size 消融实验。

这些附录的核心结论是：第一周 MVTec AD 实验证明了静态正常性记忆的有效性；第二周真实工位监控需要进一步处理位置变化、光照漂移、视频时序和人工反馈，因此应从静态 memory bank 走向动态正常性记忆。

## 参考资料

- AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2. https://arxiv.org/abs/2405.14529
- DINOv2 official repository. https://github.com/facebookresearch/dinov2
- timm / pytorch-image-models. https://github.com/huggingface/pytorch-image-models
