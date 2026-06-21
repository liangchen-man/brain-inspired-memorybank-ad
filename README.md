# 类脑计算对计算机视觉的启发：从工业异常检测中的 Memory Bank 范式理解正常性记忆

**Brain-Inspired Computer Vision: Understanding Normality Memory through the Memory Bank Paradigm in Industrial Anomaly Detection**

---

## 课程背景

本项目是类脑计算课程的实验项目，探讨如何从"正常性记忆"的角度理解工业异常检测。核心思想不是先学习所有缺陷类型，而是先建立对"正常"的记忆，再用偏离正常模式的程度判断异常。

---

## 一句话主线

**从 MVTec AD 静态 memory bank 到真实工业工位动态正常性记忆。**

---

## 已实现方法

| 方法 | Memory Bank 形式 | 说明 |
|---|---|---|
| PaDiM-teaching | 统计式记忆 | 每空间位置存储均值与逆协方差，Mahalanobis 距离判定 |
| PatchCore-teaching | 代表性 patch 记忆 | 正常 patch 特征 coreset，最近邻距离判定 |
| SPADE-teaching | 样本式记忆 | 保留正常图像特征，k-NN 检索 + 局部差分定位 |
| AnomalyDINO-style | 强特征 token 记忆 | DINOv2 pretrained patch token coreset |

---

## Week 1 摘要

在 MVTec AD `carpet` 和 `bottle` 上进行教学版复现：

- 四类 memory bank 方法均达到 Pixel AUROC 0.98+
- AnomalyDINO 预训练消融：pretrained vs non-pretrained Image AUROC 0.99559 vs 0.33427，证明 DINOv2 特征关键
- PatchCore coreset 消融：2000 → 1000，memory bank 近似减半，Pixel AUROC 几乎不变（0.98151 vs 0.98130）

---

## Week 2 摘要

在真实工业工位视频上进行迁移分析：

- **Case 4（固定透析器工位）**：侧帽缺失异常在 PatchCore+ResNet18、PatchCore+DINOv2、像素可见性基线三种方法下全部无法检测。结论：**采集方案比算法更重要**——异常信号低于相机噪声地板时，更换模型不能解决问题。
- **Case 2（滤芯传送带工位）**：8 状态 temporal baseline 发现短暂人类介入和 `4→3→4→3` 可逆流程。结论：**单帧不够，需要时序状态、短期记忆、长期记忆和人工反馈**。

---

## 目录结构

```
src/             教学版实验 runner（PaDiM, PatchCore, SPADE, AnomalyDINO, Week 2 时序/可见性基线）
tools/           图表生成脚本
report/          最终报告 v2、关键详报、素材索引、P0+P1 图表
results/         实验数据（summary.csv、各消融/视频审计/时序基线 CSV）
README.md        本文件
REPORT_INDEX.md  报告与数据文件索引
ENVIRONMENT.md   实验环境配置
ERROR_LOG.md     已知错误与待解决问题记录
HANDOFF_NEW_SESSION.md  会话交接文件
.gitignore
```

---

## 复现实验说明

```bash
conda create -n braincv-ad-py310 python=3.10
conda activate braincv-ad-py310
pip install torch torchvision opencv-python scikit-learn timm

# MVTec AD（需自行下载放入 data/mvtec/）
python src/run_patchcore.py
python src/run_padim.py
# ... etc.

# Week 2 时序基线
python src/week2_case2_temporal_baseline.py
python src/week2_visibility_baseline.py
```

---

## 数据说明

**MVTec AD 数据集和课程视频数据不随仓库发布，需用户自行准备。**

- MVTec AD：从 [MVTec 官网](https://www.mvtec.com/company/research/datasets/mvtec-ad) 下载后放入 `data/mvtec/`
- 工业工位视频：课程提供，不在本仓库分发

---

## 当前状态

**项目已进入最终报告阶段，不建议继续跑 GPU 实验。**

瓶颈在采集可观测性和时序语义理解，不在模型大小或算力。已完成实验在 RTX 3050 4GB 上均可运行。
