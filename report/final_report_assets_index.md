# 最终报告素材索引 (Final Report Assets Index)

**Date**: 2026-06-21
**Purpose**: 集中管理最终报告所需的全部图表、表格和流程图，标注完成状态和在报告中的位置
**Status**: 持续更新中

---

## 一、P0 必须图表（不做报告不完整）

### P0-1 第一周四算法热力图对比

| 字段 | 内容 |
|------|------|
| **文件路径** | `report/final_assets/fig_week1_four_algorithms_comparison.png` |
| **报告章节** | 第3章：第一周 MVTec AD 实验 — 第3.3节"实验结果要点" |
| **说明文字草稿** | "图X: 第一周四类 memory bank 在 MVTec AD carpet 缺陷图像 #000 上的异常热力图对比。从左至右：原始缺陷图像、PaDiM（统计式记忆，Mahalanobis距离）、PatchCore（代表性patch记忆，最近邻距离）、SPADE（样本式记忆，k-NN检索+局部差分）、AnomalyDINO（强特征token记忆，DINOv2 pretrained NN）。四类方法均成功定位异常区域（图中亮色高亮），在受控条件下验证了'建立正常性记忆→检测偏离'的可行性。" |
| **状态** | ✅ 已完成 |
| **Codex复核** | 建议复核：图像清晰度(150dpi)、CJK字体渲染(已用Microsoft YaHei)、算法标签准确性 |

### P0-2 Case 4 三方法失败对比表

| 字段 | 内容 |
|------|------|
| **文件路径** | `report/final_assets/table_case4_three_method_failure.md` |
| **报告章节** | 第6章：Case 4 采集可观测性限制 — 第6.2节"三方法配置与结果" |
| **说明文字草稿** | "表X: Case 4 三方法配置与检测结果对比。v1 (PatchCore+ResNet18宽ROI)、v2 (PatchCore+DINOv2紧侧帽ROI)、v3 (像素可见性基线L2/SSIM/Edge/Hist)在4个已知异常时间戳上全部失败。Top-5命中率均为0/7，v3最高z-score仅1.1（远低于z=3工业阈值）。三种独立方法论收敛于同一结论：侧帽缺失异常在当前采集条件下无法被稳定检测——瓶颈不在模型能力，在采集信号可观测性。" |
| **状态** | ✅ 已完成（Markdown表，含4张分表） |
| **Codex复核** | 建议复核：v3 z-score计算公式、正常mu/sigma来自哪些帧、z=3阈值的引用来源 |

### P0-3 静态 Memory Bank → 动态 Memory 总览流程图

| 字段 | 内容 |
|------|------|
| **文件路径** | `report/final_assets/fig_static_to_dynamic_memory.md` |
| **报告章节** | 第5章（第二周迁移概述）或第8章（类脑计算讨论）或第12章（总结） |
| **说明文字草稿** | "图X: 从MVTec AD静态正常性记忆到真实工位动态记忆的完整演化路径。第一周（绿）：MVTec AD受控条件下建立四类静态memory bank并取得AUROC 0.98+。第二周（黄/橙）：真实工位暴露出两个独立瓶颈——Case 4揭示采集可观测性限制（信号低于噪声地板），Case 2揭示单帧缺少时序上下文（需要状态机+短期记忆+人工反馈）。综合（紫）：类脑三层记忆框架将长期记忆(static bank)、短期记忆(sliding window)和反馈门控(human confirmation)整合为统一的动态正常性记忆系统。" |
| **状态** | ✅ 已完成（Mermaid源码，需渲染为PNG后放入报告） |
| **Codex复核** | 建议复核：流程图逻辑完整性、Mermaid渲染效果、Case 4/Case 2在图中是否并列正确 |

---

## 二、P1 强烈建议图表

### P1-1 Case 2 人类介入 Contact Sheet

| 字段 | 内容 |
|------|------|
| **文件路径** | 待制作 `report/final_assets/fig_case2_human_intervention_contact_sheet.png` |
| **报告章节** | 第7.3章：Case 2 突破发现 — 人类介入 |
| **说明文字草稿** | "图X: Case 2 t=197-435s关键帧接触表。红色边框标注HUMAN_INTERACTION窗口（t=389-397s，4帧）：t=389.4s操作员出现在防护板后（蓝色洁净服），t=391.3s手靠近传送带，t=393.3s手持第4支滤芯放入导轨（rail_mean跳变Δ=10，帧差=26.9），t=395.3s人类离开恢复自动化。前（t=197.6s）后（t=434.8s）各一帧正常自动化帧作为对照。人类窗口仅约4秒，在2秒采样间隔下仅2-3帧捕获——解释了初始7帧粗审计（均匀分布在472s上）为何全部未发现。" |
| **状态** | ✅ 已完成 |
| **Codex复核** | 建议复核：是否需要从视频精确提取这4帧；qwen3描述是否可作为图注替代 |

### P1-2 类脑三层记忆框架图

| 字段 | 内容 |
|------|------|
| **文件路径** | `report/final_assets/fig_brain_inspired_three_layer_memory.png` |
| **报告章节** | 第8章：类脑计算讨论 |
| **说明文字草稿** | "图X: 类脑三层正常性记忆框架。长期记忆（蓝色，新皮层/海马体巩固后的稳定经验）对应MVTec AD训练的静态memory bank（PaDiM/PatchCore/SPADE/AnomalyDINO）和Case 2的周期统计、模式库和版本化bank。短期记忆（黄色，前额叶工作记忆）对应Case 2的滑动窗口状态平滑（-66.0%假切换）、前后帧一致性检查和空位语义消歧。反馈门控（红色，前额叶-海马体交互）通过人工确认决定新状态是'合法正常扩展'还是'异常'——确认正常则巩固入长期记忆（绿色箭头），确认异常则纳入异常案例库（红色箭头）。两种失效模式分别标注：Case 4静默失效（信号<噪声）和Case 2嘈杂失效（合法变化=高分）。" |
| **状态** | ✅ 已完成 |
| **Codex复核** | 建议复核：呈现风格（手绘/方块流程图/脑区对照图）、脑区标签准确性 |

### P1-3 Case 2 单帧为什么不够 — 五层论证图解

| 字段 | 内容 |
|------|------|
| **文件路径** | `report/final_assets/fig_case2_five_level_argument.png` |
| **报告章节** | 第7.5章：为什么单帧检测不够 |
| **说明文字草稿** | "图X: Case 2 单帧检测失败的五层论证图解。从左至右逐层递进：(1) 像素-物体解耦——同一像素在不同时间对应不同物体，静态空间记忆失效。(2) 双重语义——空位可能是'正常未到达'或'异常丢失'，单帧无法区分。(3) 人手=极度异常但合法——t=393s手持滤芯在单帧上被PatchCore标记为最高分，但实际是正常补料，只有时序上下文可以抑制。(4) 正常=一组时序合法状态——4⇄3可逆是正常模式，不是固定外观分布。(5) 状态机约束=免费检测信号——非预期状态转换本身就是异常，无需训练数据。每层红色标题栏标注论证要点，黄色框写核心主张，白框写实证依据。" |
| **状态** | ✅ 已完成 |
| **Codex复核** | 建议复核：是否需要引用具体帧作为每个panel的配图 |

---

## 三、已有素材（可直接复用）

### 3.1 报告文件

| 文件 | 可放入章节 | 用途 |
|------|-----------|------|
| `report/memory_bank_comparison.md` | 第3.2章 | 四类memory bank形式概念对比 |
| `report/ablation_anomalydino_pretraining.md` | 第3.4章 | AnomalyDINO预训练消融 |
| `report/ablation_patchcore_coreset.md` | 第3.4章 | PatchCore coreset消融 |
| `report/week2_case4_failure_analysis.md` | 第6章 | Case 4完整失败分析 |
| `report/week2_visibility_baseline.md` | 第6.2章 | Case 4 v3像素可见性基线 |
| `report/week2_case2_temporal_baseline.md` | 第7章 | Case 2 8状态temporal baseline |
| `report/week2_synthesis.md` | 第5/8/12章 | Week 2双案例综合结论 |
| `report/frontier_methods_comparison.md` | 第9章 | 前沿方法对照 |
| `report/final_report_v2_outline.md` | — | 12章报告大纲（参考） |
| `report/final_report_integration_checklist.md` | — | 完整整合检查清单 |

### 3.2 已有图表文件

| 文件 | 可放入章节 | 说明 |
|------|-----------|------|
| `runs/padim_carpet/overlays/` (30张) | 第3.3章 | PaDiM热力图示例 |
| `runs/patchcore_carpet/overlays/` (30张) | 第3.3章 | PatchCore热力图示例 |
| `runs/spade_carpet/overlays/` (30张) | 第3.3章 | SPADE热力图示例 |
| `runs/anomalydino_carpet/overlays/` (30张) | 第3.3章 | AnomalyDINO热力图示例 |
| `report/figures/week2_case2_temporal_baseline/case2_frame_diff_states.png` | 第7.2章 | Case 2帧差+8色状态带 |
| `report/figures/week2_case2_temporal_baseline/case2_brightness_transition.png` | 第7.3章 | Case 2亮度过渡+人类窗口 |
| `report/figures/week2_case2_temporal_baseline/case2_state_machine.png` | 第7.2章 | Case 2 8节点状态机 |
| `report/figures/week2_case2_temporal_baseline/case2_spike_suppression.png` | 第7.2章 | Case 2尖峰抑制对比 |

### 3.3 数据文件

| 文件 | 可放入章节 | 用途 |
|------|-----------|------|
| `results/summary.csv` (7行) | 第3.4章 | 第一周四算法+两消融结果总表 |
| `results/week2/case2_temporal_baseline.csv` (240行) | 第7.2章 | Case 2完整时序标注 |
| `results/week2/case4_visibility_baseline.csv` (51行) | 第6.2章 | Case 4 v3逐透析器指标 |

---

## 四、表格清单（需放入报告）

| # | 表名 | 来源 | 报告章节 | 状态 |
|---|------|------|---------|------|
| 1 | 四算法MVTec AD结果总表 | `results/summary.csv` | 3.4 | OK 可转置为报告表格 |
| 2 | 四类memory bank概念对比表 | `report/memory_bank_comparison.md` 表1 | 3.2 | OK |
| 3 | AnomalyDINO预训练消融 | `report/ablation_anomalydino_pretraining.md` | 3.4 | OK |
| 4 | PatchCore coreset消融 | `report/ablation_patchcore_coreset.md` | 3.4 | OK |
| 5 | Case 4 v1/v2/v3配置与结果对比 | `report/final_assets/table_case4_three_method_failure.md` **新创建** | 6.2 | ✅ |
| 6 | Case 4 v3 z-score矩阵 | `report/final_assets/table_case4_three_method_failure.md` 表3 **新创建** | 6.2 | ✅ |
| 7 | Case 4 失败根因诊断 | `report/final_assets/table_case4_three_method_failure.md` 表4 **新创建** | 6.3 | ✅ |
| 8 | Case 2 8状态分布(raw vs smoothed) | `report/week2_case2_temporal_baseline.md` §3 | 7.2 | OK |
| 9 | Case 2 事件统计 | `report/week2_case2_temporal_baseline.md` §4 | 7.2 | OK |
| 10 | Case 2 单帧vs短期记忆对比 | `report/week2_case2_temporal_baseline.md` §6 | 7.5/8 | OK |
| 11 | 静态memory bank vs 长期时序memory | `report/week2_case2_temporal_baseline.md` §7.2 | 8.2 | OK |
| 12 | 三层记忆框架 | `report/week2_synthesis.md` §5 | 8.1 | OK |
| 13 | 前沿方法对照 | `report/frontier_methods_comparison.md` **之前创建** | 9 | ✅ |

---

## 五、完成度总览

| 类别 | 总数 | 已完成 | 待制作 |
|------|------|--------|--------|
| P0 必须图表 | 3 | **3** ✅ | 0 |
| P1 强烈建议 | 3 | **3** ✅ | 0 |
| 报告表格 | 13 | 5 新创建 ✅ | 8 需从已有报告提取 |
| 已有素材（可直接复用） | 8 报告 + 8 图表 + 3 数据 | 全部可用 | — |

---

## 六、下一步建议

1. **Codex 确认 P0 三件** — 浏览 `report/final_assets/` 下的三个文件，确认内容、标签、指标无错误
2. **决定 P1 制作** — 是否需要制作人类介入 contact sheet、类脑框架图、五层论证图解
3. **Mermaid 渲染** — 将 `fig_static_to_dynamic_memory.md` 中的 Mermaid 代码渲染为 PNG（可用 mermaid-cli 或在线渲染工具）
4. **开始写报告正文** — P0 图表已齐，可以按 `final_report_v2_outline.md` 开始扩写正式报告

---

*本索引文件持续更新。每新增一个素材后更新对应条目的状态。*
