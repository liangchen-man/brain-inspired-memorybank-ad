# External Repositories -- Run Notes

**Checked**: 2026-06-14 10:26
**Checked by**: DeepSeek executor (read-only)
**Method**: `ls external/` + full E-drive `find`; no training run, no code modification

---

## 1. Repository Status Summary

| Repository | Path | Exists | README | Entry Script | Supports MVTec | Outputs Heatmap | Needs Codex |
|------------|------|--------|--------|--------------|-----------------|------------------|-------------|
| spade-pytorch | `external/spade-pytorch/` | NOT FOUND | N/A | N/A | N/A | N/A | Yes |
| patchcore-inspection | `external/patchcore-inspection/` | NOT FOUND | N/A | N/A | N/A | N/A | Yes |
| AnomalyDINO | `external/AnomalyDINO/` | NOT FOUND | N/A | N/A | N/A | N/A | Yes |
| PaDiM | `external/padim/` or similar | NOT FOUND | N/A | N/A | N/A | N/A | Yes |

Root cause: `E:\leinaozuoye\external\` exists as an empty directory. No repos cloned yet.

---

## 2. Per-Repository Details

### 2.1 spade-pytorch

| Item | Value |
|------|-------|
| Path | `external/spade-pytorch/` |
| Exists | No |
| README path | N/A |
| Purpose | 需要 Codex 确认 |
| Main entry script | N/A |
| Install commands (README) | N/A |
| Python / PyTorch / CUDA requirements (README) | N/A |
| MVTec AD data path parameter | N/A |
| How to run carpet only | N/A |
| How to save heatmap / anomaly map | N/A |
| Default output location | N/A |
| Supports single-class eval | N/A |
| Windows support | N/A |
| Pretrained weights needed | N/A |
| Needs internet download | N/A |
| Conflict with braincv-ad-py310 | 需要 Codex 确认 |
| Unclear | Everything -- repo not cloned |

### 2.2 patchcore-inspection

| Item | Value |
|------|-------|
| Path | `external/patchcore-inspection/` |
| Exists | No |
| README path | N/A |
| Purpose | 需要 Codex 确认 |
| Main entry script | N/A |
| Install commands (README) | N/A |
| Python / PyTorch / CUDA requirements (README) | N/A |
| Faiss dependency | 需要 Codex 确认 |
| MVTec AD data path parameter | N/A |
| How to run carpet only | N/A |
| How to save heatmap / anomaly map | N/A |
| Default output location | N/A |
| Supports single-class eval | N/A |
| Windows support | N/A |
| Pretrained weights needed | N/A |
| Needs internet download | N/A |
| Conflict with braincv-ad-py310 | 需要 Codex 确认 |
| Unclear | Everything -- repo not cloned |

### 2.3 AnomalyDINO

| Item | Value |
|------|-------|
| Path | `external/AnomalyDINO/` |
| Exists | No |
| README path | N/A |
| Purpose | 需要 Codex 确认 |
| Main entry script | N/A |
| Install commands (README) | N/A |
| Python / PyTorch / CUDA requirements (README) | N/A |
| DINOv2 dependency | 需要 Codex 确认 |
| MVTec AD data path parameter | N/A |
| How to run carpet only | N/A |
| How to save heatmap / anomaly map | N/A |
| Default output location | N/A |
| Supports single-class eval | N/A |
| Windows support | N/A |
| Pretrained weights needed | N/A |
| Needs internet download | N/A |
| Conflict with braincv-ad-py310 | 需要 Codex 确认 (may need separate env braincv-dino-py310) |
| Unclear | Everything -- repo not cloned |

### 2.4 PaDiM

| Item | Value |
|------|-------|
| Path | `external/padim/` or similar |
| Exists | No |
| README path | N/A |
| Purpose | 需要 Codex 确认 |
| Known sources | `amazon-science/patchcore-inspection` (bundled) or `openvinotoolkit/anomalib` |
| Main entry script | N/A |
| Install commands (README) | N/A |
| Python / PyTorch / CUDA requirements (README) | N/A |
| MVTec AD data path parameter | N/A |
| How to run carpet only | N/A |
| How to save heatmap / anomaly map | N/A |
| Default output location | N/A |
| Supports single-class eval | N/A |
| Windows support | N/A |
| Pretrained weights needed | N/A |
| Needs internet download | N/A |
| Conflict with braincv-ad-py310 | 需要 Codex 确认 |
| Unclear | Everything -- repo not cloned + no standalone repo identified |

---

## 3. Blocking Items (all repos)

1. All 4 repos not cloned -- cannot read README
2. Cannot verify install commands, entry scripts, dataset paths, heatmap output
3. Cannot confirm MVTec AD compatibility from source
4. PaDiM has no standalone repo -- source must be decided
5. fg_mask.rar contents unknown -- may affect ground_truth layout

---

## 4. Next Steps

1. Codex confirms correct GitHub URLs for all 4 repos
2. Clone repos into `external/`
3. Re-run this check to populate all 15 fields per repo from actual README
4. Decide PaDiM source (anomalib vs patchcore-inspection bundled)
5. Extract fg_mask.rar to inspect internal structure
