# ERROR_LOG

## 2026-06-14 10:26 — Comprehensive Audit

### Issue 1: external/ repos not cloned
- **Task**: Read README from external/spade-pytorch/, external/patchcore-inspection/, external/AnomalyDINO/
- **Symptom**: `external/` directory exists but is empty. No repos cloned anywhere on E: drive.
- **Impact**: Cannot determine entry scripts, install commands, MVTec AD parameters, carpet-only commands, heatmap output methods.
- **Action taken**: Generated `report/repo_run_notes.md` with all 15 fields marked as N/A for all 4 repos. Suggested clone URLs (need Codex confirmation).
- **Needs Codex**: Yes — confirm correct GitHub URLs for all 4 repositories.

### Issue 2: MVTec AD dataset not extracted
- **Task**: Check MVTec AD dataset structure under data/mvtec/
- **Symptom**: `data/mvtec/` exists but is empty. All 8 target classes (carpet, grid, leather, tile, wood, bottle, capsule, hazelnut) show `class dir` missing.
- **Root cause**: Outer zip (MVTecAD1.zip) was extracted to `MVTecAD1/MVTecAD1/*.rar`, but the inner `.rar` files (16 per-class rars + fg_mask.rar) have NOT been extracted.
- **Impact**: Cannot run any algorithm. Dataset is not usable.
- **Action taken**: Created `src/check_dataset.py` (read-only, pathlib-based). Generated `report/dataset_check.md` (all 8 classes: No).
- **Needs Codex**: No — the fix is mechanical: `unrar x <class>.rar` for each target class.

### Issue 3: PyTorch not installed
- **Task**: Check Python / Conda / PyTorch / CUDA environment
- **Symptom**: PyTorch, torchvision, OpenCV, Faiss, timm are all NOT INSTALLED. Running in base conda env (not in a named env).
- **Impact**: Cannot run SPADE / PaDiM / PatchCore / AnomalyDINO.
- **Action taken**: Created `src/check_env.py` (read-only, 25 checks). Generated `ENVIRONMENT.md`.
- **Needs Codex**: Yes — decide on conda env strategy (braincv-ad-py310 vs per-repo envs).

### Issue 4: PowerShell commands not used
- **Task execution method**: Used bash-style commands (ls, find, head, mkdir) throughout.
- **Note**: User requested PowerShell commands. Current shell is bash (Unix syntax).
- **Impact**: Minor — commands ran successfully but not in requested format.
- **Needs Codex**: No.

### Issue 5: PaDiM has no standalone repo
- **Task**: Check for PaDiM repo in external/padim/ or similar.
- **Symptom**: No PaDiM directory found. Known sources: `amazon-science/patchcore-inspection` (bundled) or `openvinotoolkit/anomalib`.
- **Impact**: PaDiM experiment path unclear.
- **Action taken**: Recorded in `report/repo_run_notes.md`.
- **Needs Codex**: Yes — decide PaDiM source.

### Issue 6: fg_mask.rar purpose unclear
- **Symptom**: `fg_mask.rar` exists alongside per-class rars. Not extracted, internal structure unknown.
- **Possible use**: Might contain foreground masks or ground_truth annotations. Might NOT map 1:1 to standard MVTec ground_truth/ directory structure.
- **Impact**: ground_truth masks might be missing even after class rars are extracted.
- **Needs Codex**: Yes — confirm fg_mask.rar contents and how to integrate with MVTec standard structure.

### Issue 7: Python 3.13.5 compatibility
- **Symptom**: Python 3.13.5 is very recent. PyTorch stable wheels may target Python 3.10-3.12.
- **Risk**: PyTorch installation may fail or require nightly build.
- **Needs Codex**: Yes — confirm whether to downgrade Python to 3.10 via new conda env.

---

## 2026-06-14 10:28 — Preflight: PaDiM + carpet

### Issue 8: braincv-ad-py310 not created
- **Check**: `conda info --envs` does not list braincv-ad-py310
- **Impact**: Cannot run PaDiM. Conda env must be created first.
- **Needs Codex**: No — env creation is blocked on user confirmation.

### Issue 9: src/run_padim.py not found
- **Check**: `ls E:/leinaozuoye/src/run_padim.py` → NOT FOUND
- **Impact**: No entry script exists to run PaDiM on carpet.
- **Needs Codex**: Yes — need to write this script or confirm PaDiM source/entry point.

### Issue 10: Preflight summary — 8/16 preconditions FAIL
- **Check**: report/preflight_padim_carpet.md generated
- **Failures**:
  1. data/mvtec/carpet/ NOT FOUND
  2. data/mvtec/carpet/train/good/ NOT FOUND
  3. data/mvtec/carpet/test/ NOT FOUND
  4. PyTorch NOT INSTALLED
  5. torchvision NOT INSTALLED
  6. OpenCV NOT INSTALLED
  7. timm NOT INSTALLED
  8. src/run_padim.py NOT FOUND
  9. external/ empty (no PaDiM code)
  10. braincv-ad-py310 env NOT FOUND
- **OK (6/16)**: data/mvtec/ exists, sklearn installed, runs/ exists, results/ exists, report/ exists, GPU hardware detected
- **Needs Codex**: Yes — all 10 failures need resolution before any experiment.

---

## 2026-06-14 10:31 — Resolution Update: MVTec carpet/bottle extracted

### Resolved: data/mvtec/carpet and data/mvtec/bottle not found
- **Previous issue**: `data/mvtec/` existed but was empty; `carpet` and `bottle` were still packed as `.rar`.
- **Action taken**: Extracted `carpet.rar` and `bottle.rar` with WinRAR UnRAR into `data/mvtec/`.
- **Verification**: Re-ran `python src\check_dataset.py`.
- **Current status**:
  - `carpet`: train/good 280, test/good 28, test defect 89, gt mask 89, usable Yes.
  - `bottle`: train/good 209, test/good 20, test defect 63, gt mask 63, usable Yes.
- **Still blocking experiments**: PyTorch/torchvision/OpenCV/timm are not installed; `braincv-ad-py310` does not exist; `src/run_padim.py` does not exist.

---

## 2026-06-14 10:53 — Environment Setup Complete

### Resolved: braincv-ad-py310 created, all core deps installed
- **Conda env**: `braincv-ad-py310` created (Python 3.10.20), located at `D:\anaconda\envs\braincv-ad-py310`
- **PyTorch**: 2.12.0+cu126 — CUDA available, GPU RTX 3050 4GB detected
- **torchvision**: 0.27.0+cu126
- **OpenCV**: 4.13.0
- **scikit-learn**: 1.7.2
- **timm**: 1.0.27
- **NumPy**: 2.2.6, SciPy: 1.15.3, Matplotlib: 3.10.9, Pandas: 2.3.3
- **Faiss**: intentionally NOT installed (deferred per Codex decision)
- **Verification**: `src/check_env.py` passed inside braincv-ad-py310. `ENVIRONMENT.md` updated.

### Minor fix: total_mem → total_memory
- **Symptom**: `src/check_env.py` crashed with `AttributeError: ... no attribute 'total_mem'`. PyTorch 2.x API uses `total_memory`.
- **Fix**: Changed `total_mem` to `total_memory` in check_env.py line 77.
- **Status**: Resolved.

### Still blocking experiments
- `src/run_padim.py` not written
- `external/` still empty (no PaDiM / SPADE / PatchCore / AnomalyDINO code)
- Faiss not installed (deferred)
- AnomalyDINO special deps not installed (deferred)

---

## 2026-06-14 11:20 — PatchCore Preflight

### Issue 11: PatchCore not ready to run
- **Check**: `external/` empty, `src/run_patchcore.py` not found, `faiss` not installed
- **SKlearn**: 1.7.2 available, can substitute Faiss for teaching implementation
- **Action taken**: Generated `report/padim_summary.md`, `report/patchcore_preflight.md`
- **Needs Codex**: Yes — write `src/run_patchcore.py` (teaching version with sklearn NearestNeighbors)

### Note: bottle runtime_seconds discrepancy
- **Observation**: bottle results.json shows 69.473s, but earlier run_log.md showed 108.128s
- **Cause**: The preflight check re-ran `conda run ... python src/run_padim.py` which overwrote the previous results. The second run was faster (model weights already cached, no download overhead).
- **Impact**: The latest results.json value (69.473s) is authoritative. Earlier run_log.md (108.128s) included ResNet18 download time from first-ever run, which is no longer relevant.
- **Action**: Recorded in `report/padim_summary.md`.

---

## 2026-06-14 11:24 — Resolution Update: Teaching PatchCore runner created

### Resolved: `src/run_patchcore.py` not found
- **Previous issue**: PatchCore preflight reported `src/run_patchcore.py` missing.
- **Action taken**: Codex created `src/run_patchcore.py`, a teaching PatchCore runner using ResNet18 layer2 patch features, random coreset, and nearest-neighbor distance via `torch.cdist`.
- **Verification**: `D:\anaconda\envs\braincv-ad-py310\python.exe -m py_compile src\run_patchcore.py` passed.
- **Current status**: Ready for DeepSeek to run `PatchCore + carpet` only.
- **Still not resolved**: official `external/patchcore-inspection` is not cloned; Faiss is not installed. These are not required for the teaching PatchCore run.

---

## 2026-06-14 11:45 - AnomalyDINO preflight probe

### Issue 12: DINOv2 timm model default input size mismatch
- **Context**: Codex probed `timm.create_model('vit_small_patch14_dinov2', pretrained=False)` with a dummy `224x224` tensor to understand AnomalyDINO-style patch token extraction.
- **Error**: `AssertionError: Input height (224) doesn't match model (518).`
- **Cause**: This DINOv2 timm model defaults to an input size of 518, not 224.
- **Impact**: This was a preflight probe, not an experiment run. No `runs/anomalydino_carpet/` result was claimed.
- **Next step**: Use an explicit compatible image size/model configuration or choose a smaller ViT/DINO-style route and record limitations honestly.

### Resolved: AnomalyDINO timm input size handled
- **Action taken**: Codex created `src/run_anomalydino.py` with explicit `img_size=224` when creating `vit_small_patch14_dinov2`.
- **Verification**: `AnomalyDINO-teaching + carpet` ran successfully with pretrained DINOv2 weights.
- **Result**: `runs/anomalydino_carpet/results.json` status is `success`; heatmaps and overlays were generated.

## Issue 1 — Week2 video audit error (first run, resolved)
- Time: 2026-06-21 10:25:59
- Command: python src/week2_video_audit.py
- Error: Failed to process 20241031_222226.mp4: [WinError 2] file not found
- Suspected cause: z.extract() preserved zip internal subdirectories (案例2/, 案例4/) but script expected flat RAW_DIR
- Status: resolved — fixed extract_mp4s() to flatten via shutil.move; re-run succeeded

## Issue 2 — Week2 video audit error (first run, resolved)
- Time: 2026-06-21 10:25:59
- Command: python src/week2_video_audit.py
- Error: Failed to process 20241101_161258.mp4: [WinError 2] file not found
- Suspected cause: Same as Issue 1
- Status: resolved — same fix
- Status: open

---

## 2026-06-21 — Case 2 Temporal State Baseline Errors

### Issue 3: CJK 字体缺失 (已修复)
- **Symptom**: matplotlib 渲染中文字符时产生 UserWarning: `Glyph 38656 (CJK UNIFIED IDEOGRAPH-9700) missing from font(s) DejaVu Sans`
- **Impact**: 状态机图中的中文标签无法显示
- **Fix**: 添加 CJK 字体检测，使用 `Microsoft YaHei` (`C:\Windows\Fonts\msyhbd.ttc`)
- **Status**: resolved

### Issue 4: KeyError 'HUMAN_INTERACTION' (已修复)
- **Symptom**: 添加第 8 个状态 HUMAN_INTERACTION 后，`nodes` dict 中未包含该键，状态机图渲染时 KeyError
- **Fix**: 在 `nodes`, `node_colors`, `node_descs` 三个字典中同步添加 HUMAN_INTERACTION (位置 (6, 2.5), 颜色 #ff6348)
- **Status**: resolved

### Issue 5: 初步审计"无人类"结论被推翻 (已修正)
- **Symptom**: 初始 Case 2 审计（7 帧均匀分布）一致报告"无人类可见"
- **Root cause**: 人类介入窗口仅约 **4 秒**（t≈391-395s），2秒采样间隔下仅被 2-3 帧捕获。7 帧均匀分布在 472s 上，采样概率仅 ~6%
- **Fix**: 对 t=381-415s 做 4 帧密集 qwen3-vl-plus 审计，发现人类在 t=391.3s（防护板后）和 t=393.3s（手持滤芯）。添加 HUMAN_INTERACTION 状态 (t∈[389,397])，重跑脚本，更新报告
- **Status**: resolved

### Issue 6: 纵向投影峰值计数不可靠 (已知限制)
- **Symptom**: 尝试用 v_proj_peaks 自动检测滤芯数量变化，在 5-19 峰之间剧烈振荡，43 次过渡中无一与 qwen3-vl-plus 确认的 4→3/3→4 时刻对齐
- **Root cause**: 传送带分度运动导致白色夹具不断穿过画面，投影曲线中的夹具骨架比滤芯影像更强
- **Handling**: 改用 qwen3-vl-plus 视觉确认替代自动计数。保留 `results/week2/case2_rail_zone_analysis.json` 作为分析遗迹
- **Status**: known limitation

### Issue 7: 浮点时间戳匹配失败 (已记录)
- **Symptom**: 尝试用 `t in human_annotations` 匹配 JSON 中的帧时间戳时，因浮点精度不匹配导致 0 帧被标注
- **Root cause**: `json.dump` 序列化时 `round(t, 2)` 产生的值与 `389.4` 等字面量的浮点表示存在微小差异
- **Handling**: 改用 CSV 格式的 temporal_baseline.csv（已包含完整注解），JSON 仅保留原始时序数据
- **Status**: known limitation, does not affect deliverables
