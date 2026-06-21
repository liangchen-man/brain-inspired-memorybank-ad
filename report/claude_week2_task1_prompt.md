# 给 Claude Code 的第二周第一轮任务提示词

你是本项目的执行助手，只负责体力活和可复核的数据审计，不负责最终学术判断。项目根目录是：

```powershell
E:\leinaozuoye
```

视频数据压缩包是：

```powershell
E:\下载\视频数据集.zip
```

本轮任务：只做“第二周工位异常监控视频数据审计”。不要跑 PatchCore、AnomalyDINO、PaDiM、SPADE，不要训练，不要全量抽帧，不要删除原始 zip。

## 任务目标

1. 检查 `视频数据集.zip` 内部结构。
2. 将两个 mp4 只提取到项目内：

```text
data/workstation/raw/
```

3. 读取每个视频的元数据：文件大小、时长、FPS、分辨率、总帧数、编码信息（能读到多少写多少）。
4. 对每个视频按低帧率抽取审计帧：
   - 优先 0.5 FPS；
   - 如果帧数太多，可降到 0.2 FPS；
   - 不要超过每个视频 300 张审计帧。
5. 生成每个视频的接触图 contact sheet，方便人工快速浏览。
6. 输出 CSV 和 Markdown 报告。

## 允许创建 / 修改的文件

只允许创建或更新以下路径：

```text
data/workstation/raw/
data/workstation/audit_frames/
data/workstation/manifests/
report/figures/week2_audit_contact_sheet/
report/week2_video_audit.md
results/week2/video_inventory.csv
ERROR_LOG.md
```

如果需要新增脚本，只能放在：

```text
src/week2_video_audit.py
src/week2_extract_audit_frames.py
src/week2_make_contact_sheet.py
```

不要修改第一周算法脚本：

```text
src/run_padim.py
src/run_patchcore.py
src/run_spade.py
src/run_anomalydino.py
```

## 建议 PowerShell 命令风格

先激活环境：

```powershell
conda activate braincv-ad-py310
cd E:\leinaozuoye
```

如果你写脚本，请确保路径兼容 Windows，使用 `pathlib`，文本输出 UTF-8。

## 输出文件要求

### 1. `results/week2/video_inventory.csv`

至少包含这些列：

```text
case_id,video_path,file_size_bytes,duration_seconds,fps,width,height,frame_count,codec,notes
```

如果某项读不到，写 `null`，不要编造。

### 2. `data/workstation/manifests/audit_frames.csv`

至少包含这些列：

```text
case_id,video_id,frame_path,timestamp_seconds,frame_index,width,height,source_video
```

### 3. `report/week2_video_audit.md`

包括：

- zip 内部文件列表；
- 每个视频元数据表；
- 抽帧策略；
- 每个视频实际抽取多少帧；
- contact sheet 路径；
- 初步肉眼观察项，不能编造：
  - 是否固定机位；
  - 是否像工位监控；
  - 是否有明显 ROI；
  - 是否有人手/工具遮挡；
  - 是否存在阶段变化；
  - 是否能看到疑似异常；
  - 不确定的地方写“需要 Codex / 用户确认”。

### 4. `ERROR_LOG.md`

如有错误，追加记录：

```text
## Issue XX - Week2 video audit error
- Time:
- Command:
- Error:
- Suspected cause:
- Status:
```

无错误则不要新增错误。

## 禁止事项

- 不要删除、移动或覆盖 `E:\下载\视频数据集.zip`。
- 不要全量抽取所有帧。
- 不要运行任何异常检测模型。
- 不要自己决定最终正常样本、异常标签或实验结论。
- 不要报告 AUROC、准确率等不存在的指标。
- 不要把相邻帧随机划分为训练和测试。
- 不要保存每帧热力图。

## 交付给 Codex 的摘要格式

完成后请按下面格式回复：

```text
一、完成情况
- 创建/更新了哪些文件：
- 实际执行命令：

二、视频数据概况
- zip 内文件：
- 每个视频：时长 / FPS / 分辨率 / 总帧数 / 文件大小：

三、审计帧
- 每个视频抽帧 FPS：
- 每个视频抽取帧数：
- contact sheet 路径：

四、初步观察
- 固定机位：
- ROI：
- 遮挡：
- 阶段变化：
- 疑似异常：
- 需要 Codex 确认：

五、风险和问题
- 读取失败：
- 抽帧失败：
- 可能影响后续实验的问题：
```

