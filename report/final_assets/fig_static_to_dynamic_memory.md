# 静态 Memory Bank → 动态 Memory 总览流程图

**Date**: 2026-06-21
**Usage**: 最终报告第5/8/12章 — 展示从MVTec AD静态记忆到真实工位动态记忆的完整演化路径

---

## Mermaid 流程图

```mermaid
flowchart TB
    subgraph WEEK1["第一周：MVTec AD 静态正常性记忆"]
        A["MVTec AD 正常样本\n(carpet/bottle train/good)"]
        B["四类静态 Memory Bank\nPaDiM 统计式 | SPADE 样本式\nPatchCore 代表性patch | AnomalyDINO 强特征token"]
        C["异常热力图检测\nAUROC 0.98+"]
        A -->|"特征提取 + coreset/统计"| B
        B -->|"最近邻/马氏距离"| C
    end

    subgraph WEEK2["第二周：真实工位迁移"]
        direction TB
        D["真实工位视频\n(固定相机, 1920×1080, ~30fps)"]

        subgraph CASE4["Case 4: 采集可观测性限制"]
            E1["采集质量门控\n亮度/对焦/振动/遮挡检测"]
            E2["三方法收敛失效\nv1 ResNet18 Top-5: 0/7\nv2 DINOv2 Top-5: 0/7\nv3 像素基线 z<1.5"]
            E3["结论: 采集方案 > 算法\n瓶颈不在模型大小\n在信号可观测性"]
            E1 --> E2 --> E3
        end

        subgraph CASE2["Case 2: 时序状态与动态记忆"]
            F1["时序状态识别\n8 状态规则分类器\n(frame_diff + brightness)"]
            F2["短期记忆\n滑动中值平滑\n(窗口=5帧≈10s)\n94→32 切换 (-66%)"]
            F3["长期记忆\n周期统计 / 驻留时长\n人类介入频率\n4⇄3 可逆模式"]
            F4["人工反馈门控\nHUMAN_INTERACTION\n→ 需确认,非直接报警\n→ 防污染更新"]
            F1 --> F2 --> F3 --> F4
            F4 -.->|"确认正常"| F3
        end

        D --> CASE4
        D --> CASE2
    end

    subgraph SYNTHESIS["综合：类脑三层记忆框架"]
        G1["🧠 长期记忆\nMVTec AD static bank\n+ 周期统计 + 模式库\n→ 稳定的正常性经验"]
        G2["🧠 短期记忆\n滑动窗口 + 状态平滑\n+ 前后帧时序上下文\n→ 当前工作记忆"]
        G3["🧠 反馈门控\n人工确认 + 异常案例库\n+ 版本化bank + 回滚\n→ 防污染记忆巩固"]
        G1 <-->|"上下文查询"| G2
        G2 -->|"不确定事件"| G3
        G3 -->|"确认正常"| G1
        G3 -->|"确认异常"| G4["异常案例库\n(未来工作: DMAD启发)"]
    end

    WEEK1 -->|"迁移差距\n(Transfer Gap)"| WEEK2
    CASE4 -->|"采集瓶颈"| SYNTHESIS
    CASE2 -->|"时序瓶颈"| SYNTHESIS

    style A fill:#e8f4e8,stroke:#4caf50
    style C fill:#e8f4e8,stroke:#4caf50
    style E3 fill:#fff3cd,stroke:#ffc107
    style F4 fill:#fff3cd,stroke:#ffc107
    style G1 fill:#d1ecf1,stroke:#17a2b8
    style G2 fill:#d1ecf1,stroke:#17a2b8
    style G3 fill:#d1ecf1,stroke:#17a2b8
    style SYNTHESIS fill:#e2d9f3,stroke:#6f42c1
```

## 关键路径说明

### 路径1 (Case 4): 采集质量门控
```
MVTec AD正常样本 → 静态Memory Bank → [迁移] → 真实工位视频
    → 采集质量门控 → 三方法收敛失效
    → 结论: 信号低于噪声地板 → 改进采集而非改进模型
```

### 路径2 (Case 2): 时序状态 + 人工反馈
```
MVTec AD正常样本 → 静态Memory Bank → [迁移] → 真实工位视频
    → 时序状态识别 (8 states) → 短期记忆(sliding window)
    → 长期记忆(周期统计/人类模式) → 人工反馈门控
    → 确认正常→纳入长期记忆 | 确认异常→异常案例库
```

### 路径3: 类脑整合
```
长期记忆 ←→ 短期记忆 (上下文查询)
短期记忆 → 反馈门控 (不确定事件上报)
反馈门控 → 长期记忆 (确认正常→巩固)
反馈门控 → 异常案例库 (确认异常→独立存储)
```

## 本图的关键论点

1. **MVTec AD 静态 memory bank 是有效但有限的** — AUROC 0.98+ 仅证明在受控条件下的可行性
2. **真实工位暴露了两个独立瓶颈**:
   - Case 4: 采集可观测性（信号低于噪声）
   - Case 2: 时序语义（单帧无法区分正常/异常状态）
3. **类脑三层记忆框架整合了这两个瓶颈**:
   - 长期记忆 = static bank（受控条件有效）
   - 短期记忆 = sliding window（Case 2验证: -66%假切换）
   - 反馈门控 = human confirmation（Case 2 HUMANN_INTERACTION模式）
4. **不应自动更新 memory bank** — 需要防污染机制: 人工确认 + 版本化 + 回滚

---

*此流程图可用 Mermaid 渲染器生成 PNG，或直接在支持 Mermaid 的 Markdown 编辑器中查看。*
*推荐在最终报告中使用渲染后的 PNG 版本插入正文。*
