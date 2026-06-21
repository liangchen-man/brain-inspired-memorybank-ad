# 类脑计算对计算机视觉的启发：从工业异常检测中的 Memory Bank 范式理解正常性记忆

**项目目录**：`E:\leinaozuoye`  
**报告版本**：v2  
**日期**：2026-06-21  
**实验范围**：MVTec AD 静态异常检测基线 + 真实工业工位视频迁移分析  

---

## 摘要

工业异常检测面临一个天然困难：异常样本稀少、类型开放且难以穷举。与其先学习所有可能缺陷，不如先学习“什么是正常”，再用测试样本相对于正常模式的偏离程度判断异常。本项目围绕这一思想，第一周在 MVTec AD 的 `carpet` 与 `bottle` 类别上实现并评测 PaDiM、PatchCore、SPADE 与 AnomalyDINO-style 四类教学版 memory bank 方法；第二周将该范式迁移到真实工位视频，分析静态正常性记忆在工业现场中的失效条件。

第一周结果表明，在受控 benchmark 条件下，四类 memory bank 方法均能取得较高像素级 AUROC，其中 `carpet` 上 PaDiM、PatchCore、SPADE 与 AnomalyDINO 的 Pixel AUROC 分别为 0.98616、0.98151、0.98196 与 0.98767。消融实验进一步说明，DINOv2 预训练特征是 AnomalyDINO-style 方法有效的关键；PatchCore 的 coreset 能以较小性能损失显著压缩 memory bank。

第二周实验显示，MVTec AD 上有效的静态 memory bank 不能直接等价于真实工位部署能力。Case 4 固定透析器工位中，侧帽缺失异常在 PatchCore+ResNet18、PatchCore+DINOv2 与像素可见性基线三种方法下均无法稳定检测，说明主要瓶颈是采集信号可观测性，而非模型大小。Case 2 传送带工位中，8 状态时序基线发现短暂人类介入与 `4→3→4→3` 可逆流程，证明单帧检测不足以理解真实工位状态。

因此，本项目的核心结论是：memory bank 可以视为工业视觉中的“长期正常性记忆”，但真实工业部署还需要短期时序记忆、采集质量门控、人工反馈和防污染更新机制。这一结论体现了类脑计算中长期记忆、工作记忆与反馈门控对计算机视觉异常检测的启发。

---

## 1. 实验背景与目标

传统监督分类依赖大量带标签样本，而工业缺陷检测通常不满足这一前提。一方面，真实缺陷发生频率低，难以在项目初期收集足够样本；另一方面，缺陷类型具有开放性，部署后可能出现训练阶段未见过的新型异常。因此，工业异常检测更适合采用“正常性建模”路线：用大量正常样本建立正常模式记忆，再检测输入是否偏离该记忆。

从类脑计算角度看，memory bank 范式可被理解为一种正常经验的外部化表示。人类并不会预先记住所有异常，而是在长期经验中形成对正常场景的稳定预期；当新输入无法被正常经验解释时，才产生异常感知。本项目希望通过实验回答三个问题：

1. 在受控图像基准 MVTec AD 中，不同 memory bank 形式如何建立正常性记忆？
2. 这些静态正常性记忆迁移到真实工位视频时会遇到哪些限制？
3. 如果要面向真实工业部署，memory bank 应如何扩展为包含短期记忆、长期记忆和人工反馈的动态系统？

---

## 2. 数据与实验环境

第一周使用 MVTec AD 数据集中的 `carpet` 与 `bottle` 两个类别。二者分别代表纹理类异常和物体类异常，能够覆盖课程实验中常见的两类工业视觉问题。

| Category | Train/good | Test/good | Test defect | GT mask | 状态 |
|---|---:|---:|---:|---:|---|
| carpet | 280 | 28 | 89 | 89 | 可用 |
| bottle | 209 | 20 | 63 | 63 | 可用 |

实验环境为 Windows + Anaconda，主要配置如下。

| 项目 | 配置 |
|---|---|
| Conda env | `braincv-ad-py310` |
| Python | 3.10.20 |
| PyTorch | 2.12.0+cu126 |
| torchvision | 0.27.0+cu126 |
| GPU | NVIDIA GeForce RTX 3050 4GB Laptop |
| CUDA | 可用 |
| OpenCV | 4.13.0 |
| scikit-learn | 1.7.2 |
| timm | 1.0.27 |

第二周使用课程提供的真实工位视频数据，包含 Case 2 与 Case 4 两个场景。视频均为 1920×1080、约 30 FPS 的真实工业监控视频，缺少完整像素级 ground truth，因此第二周结论均标注为候选结论，依赖视觉审计、时间戳线索和可解释基线分析，不将候选标签伪装为正式标注。

---

## 3. 第一周：MVTec AD 静态 Memory Bank 实验

### 3.1 四类正常性记忆

本项目实现了四类教学版 anomaly detection runner，均围绕“正常样本记忆库”展开，但记忆形式不同。

| 方法 | Memory bank 形式 | 保存内容 | 异常判断方式 | 适合场景 | 迁移风险 |
|---|---|---|---|---|---|
| PaDiM | 统计式记忆 | 每个空间位置的均值与逆协方差 | Mahalanobis 距离 | 固定机位、位置稳定 | 工件位置变化会误报 |
| SPADE | 样本式记忆 | 正常图像全局特征与空间特征图 | 相似正常样本检索 + 局部差异 | 正常状态少、需参考解释 | bank 大、检索成本高 |
| PatchCore | 代表性 patch 记忆 | 正常 patch 特征 coreset | 最近邻距离 | 局部缺陷、轻量部署 | 对时序流程异常不敏感 |
| AnomalyDINO | 强特征 token 记忆 | DINOv2 patch token coreset | token 最近邻距离 | 复杂场景、泛化特征 | 依赖预训练，解释稍弱 |

PaDiM 将正常特征压缩为空间位置上的高斯统计量，体现“统计式正常经验”。PatchCore 显式保存代表性正常 patch，体现“局部正常片段记忆”。SPADE 保留正常样本全局与局部特征，体现“相似正常参考”。AnomalyDINO-style 使用 DINOv2 预训练 patch token，体现“强预训练表征上的正常性记忆”。

### 3.2 主要实验结果

第一周结果均来自 `results/summary.csv` 与各 `runs/*/results.json`。

| Algorithm | Category | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory Bank |
|---|---|---:|---:|---:|---:|---:|
| PaDiM-teaching | carpet | 0.99358 | 0.98616 | 231.634s | 31.936 MB | 13,049,029 B |
| PaDiM-teaching | bottle | 0.99603 | 0.98004 | 69.473s | 31.936 MB | 13,049,029 B |
| PatchCore-teaching | carpet | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515,205 B |
| PatchCore-teaching | bottle | 1.00000 | 0.97592 | 68.712s | 31.936 MB | 515,205 B |
| SPADE-teaching | carpet | 0.97713 | 0.98196 | 209.005s | 31.936 MB | 56,272,133 B |
| AnomalyDINO-teaching | carpet | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1,029,317 B |

四类方法在 `carpet` 上均达到 Pixel AUROC 0.98 以上，说明在 MVTec AD 的受控条件下，静态正常性记忆确实能够支持异常定位。PaDiM 与 SPADE 的 memory bank 较大，但解释直观；PatchCore 的 coreset memory bank 最小，适合讨论轻量部署；AnomalyDINO-style 的表现最好，但依赖 DINOv2 预训练表征。

第一周四算法热力图对比见 `report/final_assets/fig_week1_four_algorithms_comparison.png`。该图使用同一张 `carpet` 缺陷样本，对比原图、PaDiM、PatchCore、SPADE 与 AnomalyDINO 的 overlay，直观展示了不同正常性记忆形式如何定位同一异常区域。

### 3.3 消融实验

AnomalyDINO 预训练消融用于回答：性能提升来自 ViT 结构本身，还是来自 DINOv2 预训练特征？

| Setting | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory Bank |
|---|---:|---:|---:|---:|---:|
| pretrained=True | 0.99559 | 0.98767 | 126.472s | 101.105 MB | 1,029,317 B |
| pretrained=False | 0.33427 | 0.67811 | 203.251s | 101.105 MB | 1,029,317 B |

在相同 ViT 架构、相同 token grid 与相同 memory bank 大小下，去掉预训练后 Image AUROC 从 0.99559 降到 0.33427。这说明 AnomalyDINO-style 的关键不是“用了 ViT”，而是 DINOv2 自监督预训练提供了适合异常检测迁移的特征空间。

PatchCore coreset 消融用于回答：压缩 memory bank 是否会显著损害定位能力？

| Coreset size | Image AUROC | Pixel AUROC | Runtime | GPU Mem | Memory Bank |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.95947 | 0.98151 | 215.411s | 31.936 MB | 515,205 B |
| 1000 | 0.95104 | 0.98130 | 180.701s | 31.936 MB | 259,205 B |

Coreset 从 2000 减半至 1000 后，memory bank 近似减半，运行时间下降约 16%，Pixel AUROC 几乎不变。这说明代表性 patch 记忆不是简单保存全部正常样本，而是在正常模式覆盖和部署成本之间取得折中。

### 3.4 第一周结论

第一周的重点不是比较哪个教学版算法“排名最高”，而是理解 memory bank 是一种范式：不同算法以不同方式保存正常经验，并用测试样本到正常经验的偏离程度生成异常分数。该结果构成第二周真实工位迁移的技术基线。

---

## 4. 第二周：真实工位迁移实验设计

真实工位数据与 MVTec AD 有本质区别。MVTec AD 是对齐、受控、有标注的静态图像基准；真实工位视频则存在光照波动、运动状态、人类介入、遮挡、无完整标签和流程语义。第二周实验不应被写成“把 PatchCore 用到视频上”，而应写成对静态正常性记忆边界的检验。

本项目选择两个互补场景：

| 案例 | 场景 | 核心问题 | 作用 |
|---|---|---|---|
| Case 4 | 固定透析器装配工位，侧帽缺失 | 微小静态缺陷是否可被迁移检测 | 检验采集可观测性 |
| Case 2 | 滤芯传送带工位，间歇分度 + 人类介入 | 单帧静态 memory bank 是否足够理解状态 | 检验时序记忆必要性 |

第二周采用逐步验证策略：先进行视频审计、ROI 确认和候选事件定位，再运行必要的 PatchCore/DINOv2/像素基线；对于无 ground truth 的场景，不报告伪 AUROC，而是报告排序命中、z-score、状态识别和失败原因。

---

## 5. Case 4：采集可观测性限制

### 5.1 场景与问题

Case 4 为固定透析器装配工位，画面中有 5 支透析器。候选异常为侧帽缺失，已知标注帧共 7 个（含 4 个主要异常时间戳 t=272s、t=949s、t=1047s、t=1072s 及接近异常的辅助帧 t=270s、t=1046s、t=1054s）。该场景看似适合静态 memory bank：相机固定、工件排列稳定、异常是局部零件缺失。然而侧帽在全帧中占比极低，且受玻璃反射、光照波动和微振影响明显。

### 5.2 三方法收敛失败

本项目依次验证了三种方法：

| 方法 | 输入/ROI | 结果 | 解释 |
|---|---|---|---|
| v1 PatchCore + ResNet18 | 宽 ROI，覆盖 5 支透析器 | Top-5 0/7，Top-30 1/7 | 粗特征与大 ROI 对微小侧帽缺失不敏感 |
| v2 PatchCore + DINOv2 | 紧侧帽 ROI | Top-5 0/7，Top-30 1/7 | 强特征扩大分数范围，但无法可靠排序异常 |
| v3 Visibility Baseline | 每个侧帽 40×50px 子 ROI | 最高 z≈1.1，低于 z=3 阈值 | 像素级信号被正常波动淹没 |

更完整的配置与指标见 `report/final_assets/table_case4_three_method_failure.md`。关键结论是：即使在像素级直接对准 40×50px 侧帽区域，缺帽异常的 L2/SSIM/边缘/直方图差异仍低于正常帧间波动。三种独立方法从深度特征到像素指标都无法稳定检测，说明失败不应归因于单一模型。

### 5.3 根因分析

Case 4 的根本问题是采集信号的可观测性不足。侧帽缺失对应的有效像素变化很小，正常帧间光照闪烁和相机微振已经产生足够大的扰动。特别是 T5 区域的正常 L2 波动标准差达到 26.27，使得缺帽信号落在正常噪声范围内。

因此，工业改进的第一优先级不是租用更强 GPU 或更换更大模型，而是改善采集系统：

- 缩短相机距离或提高分辨率；
- 增加侧帽专用视角；
- 使用稳定 LED 光源和偏光滤镜；
- 固定相机与工件，减少微振；
- 在模型前加入采集质量门控；
- 对被机械臂遮挡的区域使用多相机覆盖。

Case 4 给出的课程价值是：MVTec AD 上的高 AUROC 证明了 memory bank 在受控条件下有效，但真实工业中若异常信号低于采集噪声地板，任何视觉模型都无法稳定检测。采集方案比算法更重要。

---

## 6. Case 2：单帧不够，需要时间状态

### 6.1 场景与审计发现

Case 2 为滤芯传送带工位。相机固定，但滤芯沿线性导轨进行间歇分度运动。与 Case 4 的静态排列不同，Case 2 的同一像素位置在不同时间可能对应不同滤芯、空位、运动模糊或人手，因此静态 ROI 与单帧 memory bank 的基本假设不成立。

第二轮密集审计发现，初步均匀抽帧曾漏掉一个关键事件：候选人类介入窗口 t=391-395s（qwen3-vl-plus 4 帧审计确认，蓝色洁净服操作员手持滤芯进入画面，非工位日志确认）。同时，滤芯数量变化不是简单的 `4→3`，而是候选的 `4→3→4→3` 可逆序列。这说明真实工位的“正常”不是一张固定图像，而是一组随流程变化的合法状态。

### 6.2 8 状态 Temporal Baseline

本项目构建了规则驱动的 temporal baseline，不使用深度模型，仅使用帧差、亮度和时间戳信息，并结合 qwen3-vl-plus 视觉审计确认。定义 8 个候选状态：

`INIT`、`INDEXING`、`TRANSPORT`、`DWELL`、`PROCESSING`、`TRANSITION`、`HUMAN_INTERACTION`、`END_STATE`。

状态识别结果如下。

| 状态 | Raw 帧数 | Raw 占比 | Smoothed 帧数 | Smoothed 占比 |
|---|---:|---:|---:|---:|
| DWELL | 110 | 45.8% | 93 | 38.8% |
| TRANSPORT | 69 | 28.8% | 85 | 35.4% |
| TRANSITION | 34 | 14.2% | 34 | 14.2% |
| PROCESSING | 11 | 4.6% | 16 | 6.7% |
| END_STATE | 9 | 3.8% | 7 | 2.9% |
| HUMAN_INTERACTION | 4 | 1.7% | 4 | 1.7% |
| INDEXING | 2 | 0.8% | 0 | 0% |
| INIT | 1 | 0.4% | 1 | 0.4% |

滑动中值平滑窗口约 10 秒，使状态切换次数从 94 次降到 32 次，减少 66.0%。这说明短期记忆可以显著抑制尖峰噪声，同时保留人类介入这类关键事件。相关图像见 `report/figures/week2_case2_temporal_baseline/`。人类介入关键帧的 contact sheet 见 `report/final_assets/fig_case2_human_intervention_contact_sheet.png`。

### 6.3 为什么单帧检测不足

Case 2 至少从五个层面说明单帧检测不够：

1. 同一像素位置在不同时间对应不同物体，静态空间记忆失效。
2. “空位”具有双重语义，可能是正常未到达，也可能是异常缺失。
3. 人手在单帧上是极异常外观，但在流程中可能是合法补料。
4. 正常状态是一组可逆流程，例如 `4→3→4→3`，不是固定外观。
5. 状态机约束本身就是检测信号，非预期状态转换可以直接提示异常。

因此，Case 2 不应优先运行单帧 PatchCore/DINOv2。更合理的路线是帧差监控、状态机、短期时序平滑、长期周期统计和人工反馈确认。

---

## 7. 类脑计算视角：从静态 Bank 到动态正常性记忆

本项目可以被概括为一个三层记忆框架。

| 层级 | 类脑类比 | 项目对应 |
|---|---|---|
| 长期记忆 | 稳定经验与正常模式巩固 | MVTec AD 训练得到的静态 memory bank；Case 2 的周期、亮度、状态分布统计 |
| 短期记忆 | 工作记忆与上下文保持 | Case 2 滑动窗口、状态平滑、前后帧关系 |
| 反馈门控 | 人工确认与防止错误巩固 | HUMAN_INTERACTION、候选新状态确认、版本化更新 |

静态 memory bank 相当于长期正常性记忆：它记录正常样本的稳定特征，但不理解当前输入所处的流程阶段。Case 2 中的滑动窗口和状态机相当于短期记忆：它让系统根据前后帧判断当前变化是否合理。人工反馈门控则用于决定新观察是否应纳入长期记忆，防止异常样本被错误地“巩固”为正常。

这一框架也解释了两种不同失效模式。Case 4 是“静默失效”：异常信号低于噪声，系统无法把异常排序到前列。Case 2 是“噪声失效”：运动模糊、人手、亮度变化和正常流程变化都可能产生高异常分数，但其中许多是合法操作。前者需要改善采集，后者需要时序理解。

流程图见 `report/final_assets/fig_static_to_dynamic_memory.png`（Mermaid 源码见同目录 `.md` 文件）。若最终报告转为 Word/PDF，建议直接插入 PNG 版本。

---

## 8. 前沿方法对照与本项目定位

GPT Pro 建议关注 DMAD、SimpleNet、Dinomaly、AnomalyGPT、扩散模型和 EfficientAD 等前沿方向。本项目将其作为未来工作和理论对照，而不在当前范围内实现。

| 方法 | 核心思想 | 对本项目的启发 | 当前是否实现 |
|---|---|---|---|
| DMAD | 正常与异常双记忆库 | 人工确认后的异常可进入异常案例库 | 否，未来工作 |
| SimpleNet | 合成异常明确正常边界 | 正常记忆需要反例约束 | 否，未来工作 |
| Dinomaly/Dinomaly2 | Transformer 统一模型 | 从逐类 bank 走向统一表征 | 否，未来工作 |
| AnomalyGPT/VLM | 视觉语言解释异常 | 可自动化 Case 2 的语义审计 | 否，未来工作 |
| Diffusion/RADAR | 生成式正常重建/分割 | 正常性可隐式存储在生成模型中 | 否，未来工作 |
| EfficientAD | 轻量蒸馏异常检测 | 工业部署需关注延迟与模型尺寸 | 否，未来工作 |

不实现这些方法并不是忽视前沿，而是基于实验结论做出的范围控制。Case 4 已证明瓶颈在采集可观测性，换更大模型不能从根本上解决信噪比不足；Case 2 已证明核心问题在时序语义，单帧 SOTA 也无法区分合法人类介入与异常事件。当前课程项目的顶尖价值不在于堆叠更多 SOTA，而在于用实验揭示 memory bank 工业迁移的边界条件。

---

## 9. 工业部署建议

基于两周实验，本项目提出以下部署建议。

首先，应在模型前增加采集质量门控。对于 Case 4 这类微小缺陷，若光照、反射、相机微振或遮挡使缺陷不可观测，模型输出不应被解释为可靠检测结果。

其次，应根据缺陷物理尺寸设计 ROI 与相机视角。MVTec AD 中整图对齐、缺陷清晰；真实工位中微小零件可能只占几十个像素，必须通过更近视角、专用相机或多相机系统提高可观测性。

第三，对存在运动流程的工位，应优先建立状态机与短期记忆。Case 2 表明，帧差和亮度等简单信号已能识别多种状态；深度模型应作为状态内的局部检测器，而不是替代流程理解。

第四，memory bank 更新必须经过防污染机制。不能把所有新样本自动加入正常库，否则异常和偶发干扰可能被记住为正常。更合理的做法是版本化 memory bank，并让人工确认决定新状态是合法正常扩展还是异常。

最后，显卡租赁不是本项目当前瓶颈。已完成实验在 RTX 3050 4GB 上运行时间和显存均可接受；第二周失败结论来自采集和时序，而非算力不足。因此当前不建议租用 3090/4090，除非未来要训练 Dinomaly、DMAD、扩散模型等大规模模型。

---

## 10. 局限性

本项目仍有明确限制。第一，MVTec AD 只使用 `carpet` 与 `bottle` 两类，未覆盖全部 15 类，因此第一周结果主要用于概念验证，不代表全量 benchmark 排名。第二，四类算法均为教学版实现，保留核心机制但不是官方完整复现。第三，第二周真实工位缺少完整 ground truth，Case 2 与 Case 4 的异常时间戳和事件解释均应视为候选结论，需结合现场工艺记录进一步确认。第四，Case 2 的状态机基于单个视频和 240 帧抽样，阈值和状态定义需要多批次数据验证。第五，前沿方法仅做文献方向对照，未进行复现。

这些局限不削弱本项目的核心价值，因为本项目关注的是 memory bank 范式的可解释迁移分析，而不是给出工业系统最终性能保证。

---

## 11. 总结

本项目从 MVTec AD 的静态异常检测实验出发，验证了 PaDiM、PatchCore、SPADE 与 AnomalyDINO-style 四类 memory bank 方法在受控条件下建立正常性记忆的有效性。进一步通过 AnomalyDINO 预训练消融和 PatchCore coreset 消融，说明强预训练特征和代表性记忆压缩对异常检测部署具有重要作用。

第二周真实工位迁移实验揭示了静态 memory bank 的两个关键边界。Case 4 表明，当缺陷信号低于采集噪声地板时，继续更换模型或增加算力并不能根本解决问题；Case 2 表明，当工位状态依赖时间流程、人类介入和可逆操作时，单帧检测器无法理解“正常”的语义。

因此，工业异常检测中的 memory bank 不应被理解为一个孤立算法，而应被理解为正常性记忆系统的一部分。面向真实工业部署，该系统需要长期正常性记忆、短期时序上下文、采集质量门控、人工反馈和防污染更新共同工作。这正是类脑计算对计算机视觉的启发：智能系统不仅要记住正常，还要知道何时不能相信当前输入、何时需要上下文、何时需要人工反馈，以及何时不应把新经验贸然写入长期记忆。

---

## 参考文件

- `results/summary.csv`
- `report/memory_bank_comparison.md`
- `report/ablation_anomalydino_pretraining.md`
- `report/ablation_patchcore_coreset.md`
- `report/week2_case4_failure_analysis.md`
- `report/week2_visibility_baseline.md`
- `report/week2_case2_temporal_baseline.md`
- `report/week2_synthesis.md`
- `report/frontier_methods_comparison.md`
- `report/final_assets/fig_week1_four_algorithms_comparison.png`
- `report/final_assets/table_case4_three_method_failure.md`
- `report/final_assets/fig_static_to_dynamic_memory.md`

