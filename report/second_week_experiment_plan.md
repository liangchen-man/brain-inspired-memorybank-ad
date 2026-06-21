# 第二周工位异常监控实验启动方案

## 1. 当前判断

课程手册明确第二周实做场景是“工位异常监控”。本周不应简单把视频拆成独立图片后重复第一周指标，而应回答：

> 当真实工位中的正常状态具有阶段性、时间相关性和环境漂移时，静态 memory bank 何时失效，以及如何用时间状态、短期记忆、长期记忆和人工反馈提高可靠性。

视频数据位于 `E:\下载\视频数据集.zip`，只读检查结果：

| 项目 | 内容 |
|---|---|
| 压缩包大小 | 约 3.38 GB |
| 案例数量 | 2 个：案例2、案例4 |
| 视频 | `案例2/20241031_222226.mp4`，约 1.02 GB；`案例4/20241101_161258.mp4`，约 2.35 GB |
| 辅助文件 | 2 张截图，1 个 docx |
| zip 内 docx 摘要 | “管芯的样子是不是都这样” |

当前只确认了 zip 结构，尚未确认视频时长、帧率、分辨率、固定机位、异常片段和标注情况。所有结论在视频审计前均为“未验证”。

## 2. 对 GPT Pro 建议的评估

保留：

- 将第二周定义为“阶段条件化的工位场景状态异常检测与事件级报警”。
- 使用 PatchCore 作为主线，AnomalyDINO 作为强预训练特征迁移对照。
- 不训练 LSTM、3D CNN 或 Video Transformer，优先使用时间平滑、持续报警、双阈值迟滞和事件合并。
- 按时间块或完整工位周期划分 Bank Normal、Calibration Normal、Test Stream，禁止随机打散相邻视频帧。
- 保存全量逐帧分数，热力图只保存 Top-K、事件关键帧、典型误报和典型正常帧。
- 讨论静态 memory bank 的环境漂移、几何漂移、状态混合和记忆污染问题。

收敛：

- 不把 E0-E5 全部做成大工程。第二周先完成 E0-E2，并实现一个最小 E3 原型即可。
- SPADE 不再跑全视频；保留为第一周“样本式记忆”的概念参照。
- PaDiM 只作为位置稳定性诊断，可选短片段实验，不进入主线。
- 不在视频未审计前租卡。先本地低帧率抽帧和 50-200 帧冒烟测试。

## 3. 第一阶段最小可交付路线

### 阶段 A：视频审计

目标：确认任务性质和数据规模。

产出：

- `report/week2_video_audit.md`
- `results/week2/video_inventory.csv`
- `data/workstation/audit_frames/`
- `data/workstation/manifests/audit_frames.csv`

需要记录：

- 视频数量、文件大小、时长、帧率、分辨率、总帧数。
- 是否固定机位，是否有工位 ROI，是否有人手/工具遮挡。
- 更像纹理异常、物体异常、场景状态异常，还是视频事件异常。
- 是否有官方标签；如果没有，指标不能编造 AUROC。
- 是否需要隐私处理。

### 阶段 B：低帧率抽帧与 ROI 初选

目标：生成足够人工判断的总览帧，而不是全量抽帧。

建议：

- 审计抽帧：0.2-0.5 FPS。
- 第一轮正式实验：优先 1-2 FPS。
- 先固定矩形 ROI，不引入复杂检测/跟踪。

产出：

- `data/workstation/audit_frames/<case_id>/`
- `report/figures/week2_audit_contact_sheet/`
- `report/week2_roi_notes.md`

### 阶段 C：建立第二周最小数据划分

目标：让异常检测迁移实验可复现、无数据泄漏。

建议划分：

- `Bank Normal`：只用于建立 memory bank。
- `Calibration Normal`：只用于阈值、平滑窗口和误报预算。
- `Test Stream`：连续测试流，不参与建库和调参。

禁止随机打散相邻帧。必须按时间块、工位周期或人工确认片段划分。

产出：

- `data/workstation/manifests/bank_normal.csv`
- `data/workstation/manifests/calibration_normal.csv`
- `data/workstation/manifests/test_stream.csv`
- `report/week2_split_notes.md`

## 4. 第二周实验配置

### 必跑实验

| 编号 | 实验 | 目的 | 状态 |
|---|---|---|---|
| W2-E0 | PatchCore 静态 memory bank，ROI 或整帧直接迁移 | 建立视频迁移基线 | 未完成 |
| W2-E1 | ROI vs 整帧，或单一 bank vs 阶段 bank | 验证真实工位中区域/阶段约束的价值 | 未完成 |
| W2-E2 | 原始单帧分数 vs 时间平滑和持续报警 | 回答单帧检测是否足够 | 未完成 |
| W2-E3 | AnomalyDINO 同划分对照 | 验证 DINOv2 强特征的迁移价值 | 未完成 |
| W2-E4 | 误报/失败原因分类 | 分析静态 memory bank 失效 | 未完成 |

### 顶尖加分实验

| 编号 | 实验 | 目的 | 状态 |
|---|---|---|---|
| W2-A1 | 短期记忆 + 人工确认模拟 | 验证动态正常性记忆能否缓解漂移误报 | 未完成 |
| W2-A2 | 长期/短期距离二维判断 | 区分已知正常、新正常漂移和真正异常 | 未完成 |
| W2-A3 | 事件级指标：误报/小时、检测延迟、事件合并 | 从图像检测走向工业报警 | 未完成 |

## 5. 算法路线

PatchCore：

- 第二周主方法。
- 第一周已经证明 coreset 从 2000 降到 1000 时，memory bank 从 515,205 bytes 降到 259,205 bytes，pixel AUROC 几乎不变（0.98151 到 0.98130）。
- 适合讨论部署成本、局部状态异常、缺件/错位/异物。

AnomalyDINO：

- 第二主方法和迁移能力对照。
- 第一周 pretrained vs non-pretrained 消融显示 DINOv2 预训练极关键：image AUROC 0.99559 vs 0.33427，pixel AUROC 0.98767 vs 0.67811。
- 第二周用于验证强预训练特征在真实工位视频中的迁移价值。

PaDiM：

- 可选位置稳定性诊断。
- 如果视频固定机位、工件位置稳定，可短片段试跑；若位置变化明显，则用于说明统计式位置记忆的局限。

SPADE：

- 不建议跑全视频。
- 保留为第一周样本式记忆的概念参照，最多选少量帧做定性展示。

## 6. 算力策略

当前 RTX 3050 4GB 可以完成：

- 视频 zip 结构检查。
- 低帧率抽帧。
- 50-500 帧冒烟测试。
- 小型 PatchCore / AnomalyDINO 迁移。
- 时间平滑、阈值、事件合并、绘图和报告素材整理。

暂不建议租卡。租 3090 的触发条件：

- 审计后有效测试帧超过 5000-10000 帧，并且本地全量特征提取预计超过 1 小时。
- 需要 PatchCore 和 AnomalyDINO 在同一帧清单上全量对照。
- 本地出现 OOM 或单次正式运行过慢。

默认租卡建议：先租 3090 3-4 小时，不优先 4090。4090 只在帧数超过约 2 万、截止时间非常紧、或价格差很小时考虑。

## 7. 当前需要新增的最小脚本

第一轮只需要数据处理脚本，不改第一周算法：

- `src/week2_video_audit.py`：读取 zip/视频元数据，生成 inventory。
- `src/week2_extract_audit_frames.py`：低 FPS 抽帧，生成 audit frame manifest。
- `src/week2_make_contact_sheet.py`：生成接触图，便于人工选 ROI 和正常片段。

后续再新增：

- `src/week2_build_manifest.py`：根据人工选择生成 bank/calibration/test 清单。
- `src/run_patchcore_manifest.py`：让 PatchCore 读取普通帧清单，而不是 MVTec 目录。
- `src/run_anomalydino_manifest.py`：让 AnomalyDINO 读取普通帧清单。
- `src/week2_temporal_postprocess.py`：时间平滑、迟滞、事件合并。

## 8. 给 Claude Code / DeepSeek 的分工

Claude Code / DeepSeek 负责：

- 解压或只提取指定视频到 `data/workstation/raw/`。
- 读取视频元数据。
- 低帧率抽帧。
- 生成 CSV、接触图和审计报告。
- 运行 Codex 指定的命令。
- 整理日志、错误和结果表。

Codex 必须负责：

- 判断正常样本如何选择。
- 判断 ROI 和工位阶段。
- 决定 bank/calibration/test 划分。
- 设计 PatchCore / AnomalyDINO 公平对照。
- 设计时间决策和动态记忆实验。
- 判断失败原因和最终报告故事线。

## 9. 下一步

下一步只让 Claude Code 做“视频数据审计”，不要跑模型、不要全量抽帧、不要自行判断最终正常样本。

