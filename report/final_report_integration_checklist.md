# 最终报告整合检查清单

**Date**: 2026-06-21
**Purpose**: 检查最终报告 (`report/final_report.md` 或未来正式版本) 还缺什么，避免漏掉核心素材或做了不该做的事

---

## 一、必须引用的实验文件

| 文件 | 内容 | 在报告何处引用 | 状态 |
|------|------|-------------|------|
| `results/summary.csv` | 第一周四算法 + 两个消融的完整结果（7 行） | 第 3-4 章：第一周实验表格 | OK |
| `report/memory_bank_comparison.md` | 四类记忆形式的系统性对比分析 | 第 3 章：四类 memory bank 的概念区分 | OK |
| `report/ablation_anomalydino_pretraining.md` | AnomalyDINO pretrained vs non-pretrained 消融 | 第 3.4 章：预训练消融 | OK |
| `report/ablation_patchcore_coreset.md` | PatchCore coreset 2000 vs 1000 消融 | 第 3.4 章：coreset 消融 | OK |
| `report/week2_case4_failure_analysis.md` | Case 4 三方法收敛失效分析 | 第 6 章：Case 4 采集可观测性限制 | OK |
| `report/week2_visibility_baseline.md` | Case 4 像素可见性基线（五种指标，z<3） | 第 6.2 章：v3 可见性基线 | OK |
| `report/week2_case2_temporal_baseline.md` | Case 2 8 状态 temporal baseline + HUMAN_INTERACTION | 第 7 章：Case 2 时序状态与动态 memory | OK |
| `report/week2_synthesis.md` | Week 2 双案例综合结论 | 第 5 章：迁移概述 + 第 12 章：总结 | OK |
| `report/frontier_methods_comparison.md` | **新创建** — 前沿方法对照与不实现的原因 | 第 9 章：前沿方法对照 | OK |
| `report/week2_case2_review_pack.md` | Case 2 第一版审计记录（仅作历史参考） | 第 7 章：标注为"v2 已修正" | OK（需标注 v2 修正） |
| `report/week2_case4_review_pack.md` | Case 4 审计报告（ROI 建议、异常时间戳确认） | 第 6 章：引用异常时间戳确认过程 | OK（需标注候选） |

---

## 二、必须放入最终报告的图

| 图 | 来源 | 用途 | 状态 |
|----|------|------|------|
| 第一周四算法热力图对比图 | `runs/padim_carpet/overlays/` + PatchCore/SPADE/AnomalyDINO | 第 3.3 章：直观对比四类 memory bank 的检测效果 | 待整合为一页多 panel 对比图 |
| AnomalyDINO 预训练消融热力图对比 | `runs/anomalydino_carpet/` vs `runs/anomalydino_carpet_abl/` | 第 3.4 章：有/无预训练的定性差异 | 可复用已有 overlay 图 |
| Case 4 三方法失败对比表 | 从 `week2_case4_failure_analysis.md` 提取 | 第 6.2 章：v1/v2/v3 配置与结果一览 | 待制表（已有原始数据） |
| Case 4 可见性基线子 ROI 标注图 | 视频帧 + 5 个侧帽子 ROI 框 | 第 6.2 章：展示 40×50px 子 ROI 位置 | 待标注 qwen3 确认过的子 ROI 位置 |
| Case 4 可见性基线 z-score 矩阵 | `week2_visibility_baseline.md` 中的表格 | 第 6.2 章：所有异常时间戳的所有透析器 z<1.5 | 可直接复用表格 |
| Case 2 亮度 + 帧差时序图 | `report/figures/week2_case2_temporal_baseline/case2_frame_diff_states.png` | 第 7.2 章：展示 8 色状态背景带 | OK |
| Case 2 亮度过渡区 + 人类窗口标注图 | `report/figures/week2_case2_temporal_baseline/case2_brightness_transition.png` | 第 7.3 章：展示 HUMAN_INTERACTION 窗口 | OK |
| Case 2 8 节点状态机图 | `report/figures/week2_case2_temporal_baseline/case2_state_machine.png` | 第 7.4 章：展示状态转换规则（含 HUMAN_INTERACTION） | OK |
| Case 2 人类介入事件时序说明图 | 需新制——4 帧 qwen3 审计的 contact sheet 式排列 | 第 7.3 章：visual proof of human at t=391-395s | 待制（已有 4 帧描述，可复用 contact sheet 工具） |
| Case 2 4→3→4→3 可逆序列说明图 | 需新制——时间线 + 滤芯数变化 + 关键帧 | 第 7.4 章：展示可逆性 | 待制（已有 CSV 和时序数据） |
| Case 2 单帧为什么不够的图解 | 需新制——空位双重语义 + 人手正常但异常 + 4⇄3 可逆 | 第 7.5 章：五层论证的可视化 | 待制 |
| 类脑三层记忆框架图 | 需新制——长期记忆(static bank) → 短期记忆(sliding window) → 反馈门控(human confirmation) 的流程图 | 第 8 章：类脑计算讨论 | 待制 |
| 静态 memory bank → 动态 memory 的流程图 | 需新制——从 MVTec AD 到 Case 4 到 Case 2 的演化箭头 | 第 5/8/12 章：总览流程图 | 待制 |

---

## 三、必须回答的课程问题（在报告中显式回答）

| 问题 | 对应章节 | 答案要点 | 状态 |
|------|---------|---------|------|
| 为什么不先学习缺陷，而是先学习正常？ | 2 | 缺陷样本稀少且类型开放，无法枚举所有异常；先建正常模型更经济，符合工业实际 | 有素材，待写入 |
| SPADE / PaDiM / PatchCore / AnomalyDINO 的 memory bank 有什么不同？ | 3 | 统计式 vs 样本式 vs 代表性 patch vs 强特征 token，见 `memory_bank_comparison.md` | OK，需整合 |
| 静态 memory bank 为什么在真实工业现场可能失效？ | 6 + 7 | Case 4：采集信号不可观测；Case 2：缺少时序上下文；两种失效模式互补 | OK，需整合 |
| 单帧检测是否足够？ | 7.5 | Case 2 五层论证证明不够：像素-物体解耦、双重语义、人手误报、多外观正常、状态机信号 | OK |
| 是否需要短期记忆和长期记忆？ | 8 | Case 2 滑动窗口 (-66.0% 假切换) + Case 4 采集质量门控 + 两种记忆的互补角色 | OK |
| 为什么 memory bank 不能随便自动更新？ | 8.3 | 防污染——异常也可能被记住；版本化 bank + 人工确认 + 回滚能力 | 有素材，待扩写 |
| 如何设计人工反馈和防污染机制？ | 8 + 10 | 人工确认新状态是否为合法正常操作；HUMAN_INTERACTION 标记为"需确认"非直接报警 | 有素材，待扩写 |

---

## 四、必须放入报告的表格

| 表 | 来源 | 用途 | 状态 |
|----|------|------|------|
| 四算法 MVTec AD 结果总表 | `results/summary.csv` | 第 3.4 章：image/pixel AUROC + runtime + GPU mem + memory bank bytes | OK |
| 四类 memory bank 形式概念对比表 | `memory_bank_comparison.md` 第一张表 | 第 3.2 章：统计式/样本式/代表性/强特征 token | OK |
| AnomalyDINO 预训练消融对比表 | `ablation_anomalydino_pretraining.md` | 第 3.4 章 | OK |
| PatchCore coreset 消融对比表 | `ablation_patchcore_coreset.md` | 第 3.4 章 | OK |
| Case 4 v1/v2/v3 配置与结果对比表 | `week2_case4_failure_analysis.md` | 第 6.2 章：三方法收敛失败一览 | OK（需排版） |
| Case 4 可见性基线 z-score 矩阵 | `week2_visibility_baseline.md` | 第 6.2 章 | OK |
| Case 2 8 状态分布表 (raw vs smoothed) | `week2_case2_temporal_baseline.md` §3 | 第 7.2 章 | OK |
| Case 2 事件统计 (44 事件帧, 7 类型) | `week2_case2_temporal_baseline.md` §4 | 第 7.2 章 | OK |
| Case 2 单帧 vs 短期记忆对比表 | `week2_case2_temporal_baseline.md` §6 | 第 7.5 / 8 章 | OK |
| 静态 memory bank vs 长期时序 memory 对比表 | `week2_case2_temporal_baseline.md` §7.2 | 第 8.2 章 | OK |
| 三层记忆框架表 | `week2_synthesis.md` §5 | 第 8.1 章 | OK |
| 前沿方法对照表 | `frontier_methods_comparison.md` **新创建** | 第 9 章 | OK |
| Case 4 vs Case 2 互补性总表 | `week2_case2_review_pack.md` §5 + `week2_synthesis.md` §1 | 第 5 章：双案例 narrative 总表 | 待整合 |

---

## 五、不应继续做的事

- [x] **不要再跑 Case 4 深度模型** —— 三方法收敛已足够证明"采集方案 > 算法"
- [x] **不要再跑 Case 2 PatchCore/DINOv2** —— 8 状态 baseline 已足够证明"单帧不够"
- [x] **不要租显卡** —— 所有实验在 RTX 3050 4GB 上已完成，无 GPU 需求
- [x] **不要实现 DMAD / Dinomaly / 扩散模型** —— 列为未来工作，不实验
- [x] **不要全量跑 MVTec 所有 15 类别** —— carpet + bottle 已支撑 memory bank 概念
- [x] **不要编造指标** —— 不声称本项目实现了未复现的方法
- [x] **不要把候选标签写成 ground truth** —— 所有 Case 2/Case 4 异常标签标记"候选"
- [x] **不要修改已有 `src/` `runs/` `results/` `data/`** —— 只读

---

## 六、还需要 Codex 判断的问题

### 6.1 Case 2 遗留问题（来自 `HANDOFF_NEW_SESSION.md`）

- [ ] **Is 4→3→4→3 standard operating procedure?** — 如果 S.O.P.，则"可逆"进一步强化 temporal memory 论据
- [ ] **t≈380-389s 1st removal**: automated or human?
- [ ] **t≈403-405s 2nd removal**: automated or human?
- [ ] **Human intervention frequency**: once per batch normal, or this video is special?
- [ ] **Brightness transition zone root cause**: camera auto-exposure, lighting change, or equipment action?

### 6.2 报告结构问题

- [ ] **报告的正式语言**：中文 or 双语？建议中文正文 + 英文图表标题
- [ ] **文献引用格式**：IEEE / GB/T 7714 / 其他？
- [ ] **前沿方法章节的深度**：详细讨论 or 简短列举 + 引用 `frontier_methods_comparison.md`？
- [ ] **热力图排版方案**：每一类算法展示 3-5 张代表性热力图，还是用统一的多 panel 图对比？

### 6.3 图制作问题

- [ ] **人工介入 contact sheet 图**：需要从视频中提取 t=389.4, 391.3, 393.3, 395.3s 的 4 帧并排列——Codex 需确认是否需要
- [ ] **4→3→4→3 时序说明图**：用时间线 or 状态条 or 帧序列？Codex 决定呈现方式
- [ ] **类脑三层记忆框架图**：手绘风格 or 方块流程图？Codex 决定呈现方式

---

## 七、当前整合进度

| 章节 | 报告位置 | 素材状态 | 写入状态 |
|------|---------|---------|---------|
| 1. 实验题目 | 报告开头 | OK | 待写入 |
| 2. 实验背景 | 报告开头 | OK | 待写入 |
| 3. 第一周 MVTec AD | 报告主体 §1-2 | OK (summary.csv + memory_bank_comparison.md) | 待写入 |
| 4. 第一周总结 | 报告主体 §2 | OK (两个消融) | 待写入 |
| 5. 第二周迁移概述 | 报告主体 §3 | OK (week2_synthesis.md) | 待写入 |
| 6. Case 4 | 报告主体 §3.1 | OK (failure_analysis + visibility baseline) | 待写入 |
| 7. Case 2 | 报告主体 §3.2 | OK (temporal_baseline + review_pack v2修正) | 待写入 |
| 8. 类脑计算讨论 | 报告主体 §4 | OK (synthesis + temporal_baseline) | 待写入 |
| 9. 前沿方法对照 | 报告主体 §5 | **OK (frontier_methods_comparison.md 新创建)** | 待写入 |
| 10. 工业部署建议 | 报告主体 §6 | OK | 待写入 |
| 11. 局限性 | 报告结尾 | OK | 待写入 |
| 12. 总结 | 报告结尾 | OK | 待写入 |

---

## 八、图制作优先级建议

**P0 — 必须做，不做报告不完整**：
1. 第一周四算法热力图对比 (一页多 panel)
2. Case 4 三方法失败对比表
3. 静态 memory bank → 动态 memory 总览流程图

**P1 — 强烈建议**：
4. Case 2 人类介入 contact sheet (4 帧排列)
5. 类脑三层记忆框架图
6. Case 2 5 层论证图解

**P2 — 锦上添花**：
7. Case 4 子 ROI 标注图
8. Case 2 4→3→4→3 事件说明图

---

*本 checklist 不需要一次全部完成。标注"待写入"的章节可以逐步补充。*
*所有"待制"的图需要 Codex 确认呈现方式后才能制作。*
