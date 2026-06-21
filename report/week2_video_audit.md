# Week 2: Workstation Video Audit Report

**Generated**: 2026-06-21 10:41:32
**Source**: `E:/下载/视频数据集.zip`

---

## 1. ZIP Internal File List

| # | Filename | Type | Size |
|---|----------|------|------|
| 1 | 案例2/ | DIR | 0 B |
| 2 | 案例2/无标题.png | PNG image | 1.77 MB |
| 3 | 案例2/新建 DOCX 文档.docx | DOCX document | 10.23 KB |
| 4 | 案例2/20241031_222226.mp4 | MP4 video | 1.02 GB |
| 5 | 案例4/ | DIR | 0 B |
| 6 | 案例4/屏幕截图 2026-04-27 092740.png | PNG screenshot | 1.48 MB |
| 7 | 案例4/20241101_161258.mp4 | MP4 video | 2.35 GB |

**Note**: 案例2 = Case 2, 案例4 = Case 4. The zip contains 2 videos from 2 different cases. The PNG images and DOCX document are supplementary materials not used for anomaly detection.

---

## 2. Video Metadata

| Property | Video 1 (案例2) | Video 2 (案例4) |
|----------|----------------|-----------------|
| Filename | 20241031_222226.mp4 | 20241101_161258.mp4 |
| File size | 1.02 GB | 2.35 GB |
| Duration | 472.61s (7.9 min) | 1091.21s (18.2 min) |
| FPS | 29.85 | 29.95 |
| Resolution | 1920x1080 | 1920x1080 |
| Frame count | 14108 | 32685 |
| Codec | h264 (H.264) | h264 (H.264) |

Both videos: 1080p, ~30 FPS, H.264 encoded. Both are standard surveillance/industrial camera recordings.

---

## 3. Audit Frame Extraction Strategy

- **Target FPS**: 0.5 FPS (every 2 seconds)
- **Max frames per video**: 300
- **Fallback**: If estimated frames exceed max, FPS lowered proportionally

| Video | Target FPS | Effective FPS | Frames Extracted | Frame Interval |
|-------|-----------|---------------|-----------------|----------------|
| 20241031_222226 | 0.5 | 0.500 | 240 | ~2.0s |
| 20241101_161258 | 0.5 | 0.275 | 300 | ~3.6s |

Video 2 exceeded the 300-frame cap at 0.5 FPS (would produce ~546 frames), so effective FPS was lowered to 0.275.

---

## 4. Contact Sheets

- [20241031_222226 contact sheet](figures/week2_audit_contact_sheet/20241031_222226_contact_sheet.png) (2.3 MB)
- [20241101_161258 contact sheet](figures/week2_audit_contact_sheet/20241101_161258_contact_sheet.png) (2.2 MB)

---

## 5. Preliminary Visual Observations

> **Methodology**: Sampled 6 frames across both videos (early/mid/late) and analyzed via vision model (qwen3-vl-plus). Observations are visual only — no pixel-level computation or anomaly scoring was performed.

### 5.1 Video 1: 20241031_222226 (案例2) — 7.9 min

**Scene**: Automated precision assembly/inspection station in a cleanroom-like environment. White honeycomb-style acrylic partition walls. Equipment: aluminum extrusion linear conveyor, transparent cylindrical workpieces (filter cartridges or reagent tubes), pneumatic actuators, photoelectric sensors, blue air tubing.

**Key observations across time (0s → 452s)**:
| Aspect | 0s (start) | 235s (mid) | 452s (late) |
|--------|-----------|------------|------------|
| Workpiece position | Stationary, aligned on conveyor | Moving rightward on conveyor | Static, aligned |
| Sensor status | Probe sensor in place | Sensor triggered (workpiece passed) | Green indicator LED on |
| Pneumatic actuator | Retracted / standby | Retracted | Retracted |
| Human present | No | No | Yes — hand visible in upper-right corner, holding a transparent container |
| Lighting | Stable, uniform cool-white LED | Same | Same |
| Anomalies | None visible | None | Minor: hand presence suggests human intervention; unclear if authorized/gloved |

**Summary**:
- Fixed camera: **YES** — viewpoint unchanged across all frames
- Looks like workstation monitoring: **YES** — automated assembly/inspection station
- Obvious ROI: **YES** — conveyor area with workpieces and sensor probe
- Human presence: **Occasional** — one human hand visible at t≈452s
- Tool / hand occlusion: **Partially** — hand partially in frame at late stage
- Scene changes: **Yes** — workpieces move along conveyor; minor lighting/reflection variation
- Visible anomaly: **Unclear** — human hand at t≈452s may be normal operation or non-standard intervention

### 5.2 Video 2: 20241101_161258 (案例4) — 18.2 min

**Scene**: Dialyzer (hollow fiber dialyzer/HDF) automated production or inspection station. Stainless steel panels + transparent acrylic protective cover. 5–6 vertically mounted cylindrical dialyzer modules (branded "WAZOER"/"同谱", model "F16", labeled "Hollow Fiber Dialyzer"). Mechanical gripper/clamp assembly on the left side.

**Key observations across time (0s → 1042s)**:
| Aspect | 0s (start) | 267s (mid) | 808s | 1042s (late) |
|--------|-----------|------------|------|-------------|
| Dialyzer count | 5 visible | 6 visible | — | 5 visible |
| Dialyzer alignment | All aligned, leftmost appears not fully seated | — | — | All aligned |
| Gripper status | Static | — | — | Static |
| Protective cover | Present, with reflections | Same | Same | Same |
| Human present | No | No | — | No |
| Lighting | Stable cool-white LED | Same | Same | Same |
| Anomalies | Leftmost module slightly misaligned (possible assembly in progress) | — | — | No |

**Summary**:
- Fixed camera: **YES** — identical viewpoint with acrylic cover reflections consistent
- Looks like workstation monitoring: **YES** — medical device assembly/inspection station
- Obvious ROI: **YES** — the dialyzer modules on the mounting rail
- Human presence: **No human observed** in any sampled frame
- Tool / hand occlusion: **None observed**
- Scene changes: **Minor** — dialyzer count varied (5 vs 6), leftmost module position slightly different
- Visible anomaly: **Unclear** — leftmost module appeared not fully seated at t=0s; may be normal step in assembly cycle

### 5.3 Cross-Video Comparison

| Aspect | Video 1 (案例2) | Video 2 (案例4) |
|--------|----------------|-----------------|
| Industry domain | Filter cartridge / reagent tube assembly | Medical dialyzer (透析器) production |
| Environment | Cleanroom with acrylic partitions | Stainless steel + acrylic cover |
| Automation level | Fully automated conveyor + sensor | Semi-automated with gripper |
| Human interaction | Occasional (1 intervention observed) | None observed |
| Stationarity | Workpieces move (conveyor) | Workpieces mostly static |
| Potential anomaly indicators | Human hand at t≈452s | Left dialyzer misalignment at t=0s |

---

## 6. Observations Requiring Codex / User Confirmation

1. **Video 1 — Human hand at t≈452s**: Is this normal operator intervention (material replenishment, QC sampling) or unauthorized access? Need to review the DOCX document in 案例2/ for SOP context.

2. **Video 1 — Conveyor workpiece type**: Are these filter cartridges, reagent tubes, or something else? The PNG image in 案例2/ folder may provide product identification.

3. **Video 2 — Dialyzer count change (5 vs 6)**: Does this represent a normal loading/unloading cycle, or is one dialyzer missing/removed anomalously?

4. **Video 2 — Leftmost dialyzer misalignment**: Is the slightly offset module at t=0s a normal "partially loaded" state before clamping, or a defect signal?

5. **Both videos — Anomaly definition**: What constitutes an "anomaly" for these workstations? Missing workpiece, misalignment, human intrusion, illumination change, equipment stoppage?

6. **Both videos — Train/val/test split strategy**: Videos are temporal sequences. How should frames be partitioned for anomaly detection experiments? Time-based split (first N minutes = normal, later = test)?

---

## 7. Files Created

| Path | Description | Size/Rows |
|------|-------------|-----------|
| `data/workstation/raw/20241031_222226.mp4` | Video 1 (案例2) raw | 1.02 GB |
| `data/workstation/raw/20241101_161258.mp4` | Video 2 (案例4) raw | 2.35 GB |
| `data/workstation/audit_frames/20241031_222226/` | Video 1 audit frames | 240 PNG files |
| `data/workstation/audit_frames/20241101_161258/` | Video 2 audit frames | 300 PNG files |
| `data/workstation/manifests/audit_frames.csv` | Frame manifest | 540 rows |
| `results/week2/video_inventory.csv` | Video metadata inventory | 2 rows |
| `report/week2_video_audit.md` | This report | — |
| `report/figures/week2_audit_contact_sheet/` | Contact sheets (2 PNGs) | ~4.5 MB |
| `src/week2_video_audit.py` | Audit extraction script | — |
