#!/usr/bin/env python3
"""Week 2 Case 4 Review Pack Preparation.

Extracts key frames from 20241101_161258.mp4, generates ROI candidate images,
creates manual labeling template CSV, and extracts auxiliary PNG from zip.

Does NOT: run anomaly detection, make final labels, or determine normal/abnormal.
"""

from __future__ import annotations

import csv
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path("E:/leinaozuoye")
VIDEO_PATH = PROJECT_ROOT / "data" / "workstation" / "raw" / "20241101_161258.mp4"
ZIP_PATH = Path("E:/下载/视频数据集.zip")
AUDIT_CSV = PROJECT_ROOT / "data" / "workstation" / "manifests" / "audit_frames.csv"

REVIEW_DIR = PROJECT_ROOT / "data" / "workstation" / "review_frames" / "20241101_161258"
ROI_DIR = PROJECT_ROOT / "report" / "figures" / "week2_case4_roi_candidates"
MANIFEST_DIR = PROJECT_ROOT / "data" / "workstation" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / "week2_case4_review_pack.md"

# ROI candidates (x1, y1, x2, y2) — normalized 0..1, will be scaled to 1920x1080
ROI_CANDIDATES = {
    "roi1_dialyzer_only": {
        "label": "ROI 1: 透析器主体区域",
        "norm": (0.25, 0.15, 0.62, 0.88),
        "color": (0, 255, 0),  # green
    },
    "roi2_dialyzer_plus_gripper": {
        "label": "ROI 2: 透析器 + 左侧夹具区域",
        "norm": (0.08, 0.10, 0.62, 0.92),
        "color": (255, 0, 0),  # blue
    },
    "roi3_wide_workstation": {
        "label": "ROI 3: 较宽工位关键区域",
        "norm": (0.02, 0.02, 0.75, 0.96),
        "color": (0, 0, 255),  # red
    },
}


def extract_key_frames() -> list[dict]:
    """Extract key frames from video at specified time intervals."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Build set of target timestamps
    targets: set[float] = set()

    # Every 30s globally
    t = 0.0
    while t <= duration:
        targets.add(round(t, 1))
        t += 30.0

    # First 120s: every 10s
    t = 0.0
    while t <= 120.0 and t <= duration:
        targets.add(round(t, 1))
        t += 10.0

    # Last 120s: every 10s
    start_t = max(0, duration - 120)
    t = start_t
    while t <= duration:
        targets.add(round(t, 1))
        t += 10.0

    targets = sorted(targets)
    print(f"Target timestamps: {len(targets)} unique timestamps from {targets[0]:.1f}s to {targets[-1]:.1f}s")

    records = []
    target_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_t = frame_idx / fps if fps > 0 else 0.0

        while target_idx < len(targets) and current_t >= targets[target_idx] - 0.5 / fps:
            # Close enough to target — save this frame
            ts = targets[target_idx]
            fname = f"case4_t{ts:07.1f}s.png"
            fpath = REVIEW_DIR / fname
            if not fpath.exists():
                cv2.imwrite(str(fpath), frame)
            h, w = frame.shape[:2]
            records.append({
                "case_id": "case4",
                "video_id": "20241101_161258",
                "timestamp_seconds": round(ts, 1),
                "frame_index": frame_idx,
                "frame_path": str(fpath),
                "width": w,
                "height": h,
                "source_video": str(VIDEO_PATH),
            })
            saved += 1
            target_idx += 1

        if target_idx >= len(targets):
            break

        frame_idx += 1

    cap.release()
    print(f"  Saved {saved} key frames to {REVIEW_DIR}")
    return records


def generate_roi_images(records: list[dict]) -> None:
    """Draw ROI bounding boxes on 3 representative frames."""
    ROI_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 3 representative frames: early, mid, late
    if len(records) < 3:
        picks = records
    else:
        n = len(records)
        picks = [records[0], records[n // 2], records[-1]]

    for pick in picks:
        img = cv2.imread(pick["frame_path"])
        if img is None:
            continue
        h, w = img.shape[:2]

        # For each ROI, draw on a copy
        for roi_key, roi_info in ROI_CANDIDATES.items():
            nx1, ny1, nx2, ny2 = roi_info["norm"]
            x1 = int(nx1 * w)
            y1 = int(ny1 * h)
            x2 = int(nx2 * w)
            y2 = int(ny2 * h)
            color = roi_info["color"]

            img_copy = img.copy()
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
            # Draw label
            label = roi_info["label"]
            cv2.putText(img_copy, label, (x1 + 5, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            ts = pick["timestamp_seconds"]
            out_name = f"case4_t{ts:07.1f}s_{roi_key}.png"
            out_path = ROI_DIR / out_name
            cv2.imwrite(str(out_path), img_copy)
            print(f"  ROI image: {out_path}")

        # Also save original (no boxes)
        ts = pick["timestamp_seconds"]
        orig_name = f"case4_t{ts:07.1f}s_original.png"
        orig_path = ROI_DIR / orig_name
        if not orig_path.exists():
            cv2.imwrite(str(orig_path), img)
            print(f"  Original: {orig_path}")


def extract_case4_png() -> Path | None:
    """Extract the case 4 auxiliary PNG from the zip."""
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            if "案例4" in info.filename and info.filename.lower().endswith(".png"):
                basename = Path(info.filename).name
                dest = ROI_DIR / basename
                if not dest.exists():
                    content = z.read(info.filename)
                    dest.write_bytes(content)
                    print(f"Extracted auxiliary PNG: {dest}")
                else:
                    print(f"Auxiliary PNG already exists: {dest}")
                return dest
    print("WARNING: Could not find case 4 PNG in zip")
    return None


def generate_label_template(records: list[dict]) -> None:
    """Generate manual label template CSV."""
    out_path = MANIFEST_DIR / "week2_case4_manual_label_template.csv"
    headers = [
        "case_id", "video_id", "timestamp_seconds", "frame_index", "frame_path",
        "candidate_phase",
        "candidate_use",
        "manual_label",
        "manual_notes",
        "roi_candidate",
        "confirmed_by",
    ]

    rows = []
    duration = records[-1]["timestamp_seconds"] if records else 0

    for r in records:
        ts = r["timestamp_seconds"]

        # Heuristic candidate_phase — ALL marked uncertain/unknown
        if ts <= 30:
            phase = "possible_startup"
        elif ts <= duration - 30:
            phase = "possible_stable"
        else:
            phase = "possible_ending"

        # Heuristic candidate_use — ALL tentative
        if ts <= 120:
            use = "candidate_calibration"
        elif ts <= 300:
            use = "candidate_bank_normal"
        else:
            use = "candidate_test"

        rows.append({
            "case_id": r["case_id"],
            "video_id": r["video_id"],
            "timestamp_seconds": ts,
            "frame_index": r["frame_index"],
            "frame_path": r["frame_path"],
            "candidate_phase": phase,
            "candidate_use": use,
            "manual_label": "",
            "manual_notes": "",
            "roi_candidate": "待确认",
            "confirmed_by": "",
        })

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    print(f"Label template: {out_path} ({len(rows)} rows)")


def write_review_frames_csv(records: list[dict]) -> None:
    """Write review frames manifest CSV."""
    out_path = MANIFEST_DIR / "week2_case4_review_frames.csv"
    headers = [
        "case_id", "video_id", "timestamp_seconds", "frame_index",
        "frame_path", "width", "height", "source_video",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(records)
    print(f"Review frames manifest: {out_path} ({len(records)} rows)")


def write_report(
    records: list[dict],
    png_path: Path | None,
    roi_keys: list[str],
) -> None:
    """Write week2_case4_review_pack.md."""
    duration = records[-1]["timestamp_seconds"] if records else 0

    lines = [
        "# Week 2 Case 4 Review Pack",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Video**: `data/workstation/raw/20241101_161258.mp4`",
        f"**Video duration**: {duration:.1f}s ({duration/60:.1f} min)",
        "",
        "---",
        "",
        "## 1. Input Files",
        "",
        "- `data/workstation/raw/20241101_161258.mp4` — source video (case 4)",
        "- `data/workstation/manifests/audit_frames.csv` — previous audit frame manifest",
        "- `report/figures/week2_audit_contact_sheet/20241101_161258_contact_sheet.png` — contact sheet",
        "- `E:/下载/视频数据集.zip` — original zip (案例4 PNG extracted)",
        "",
        "## 2. Review Frames Extraction",
        "",
        f"- **Total frames extracted**: {len(records)}",
        f"- **Time range**: {records[0]['timestamp_seconds']:.1f}s – {records[-1]['timestamp_seconds']:.1f}s",
        f"- **Strategy**:",
        "  - Every 30s globally across full duration",
        "  - First 120s: every 10s (dense startup sampling)",
        "  - Last 120s: every 10s (dense ending sampling)",
        f"- **Output directory**: `data/workstation/review_frames/20241101_161258/`",
        "",
        "## 3. ROI Candidates",
        "",
        "ROI candidate images are in: `report/figures/week2_case4_roi_candidates/`",
        "",
        "| ROI | Label | Normalized Coords (x1,y1,x2,y2) | Pixel Coords (1920x1080) |",
        "|-----|-------|-------------------------------|--------------------------|",
    ]

    for roi_key, roi_info in ROI_CANDIDATES.items():
        nx1, ny1, nx2, ny2 = roi_info["norm"]
        px1, py1 = int(nx1 * 1920), int(ny1 * 1080)
        px2, py2 = int(nx2 * 1920), int(ny2 * 1080)
        lines.append(
            f"| {roi_key} | {roi_info['label']} | "
            f"({nx1:.2f}, {ny1:.2f}, {nx2:.2f}, {ny2:.2f}) | "
            f"({px1}, {py1}, {px2}, {py2}) |"
        )

    lines.extend([
        "",
        f"**ROI images generated** for timestamps: "
        f"{records[0]['timestamp_seconds']:.1f}s, "
        f"{records[len(records)//2]['timestamp_seconds']:.1f}s, "
        f"{records[-1]['timestamp_seconds']:.1f}s",
        "",
        "## 4. Case 4 Auxiliary PNG",
        "",
    ])
    if png_path and png_path.exists():
        lines.append(f"- **Extracted to**: `{png_path}`")
        lines.append("- **Status**: 需要 Codex / 用户确认 — 检查是否与视频场景一致，是否能帮助判断 ROI")
    else:
        lines.append("- **Status**: NOT FOUND in zip")
    lines.extend([
        "",
        "## 5. Manual Label Template",
        "",
        f"- **Path**: `data/workstation/manifests/week2_case4_manual_label_template.csv`",
        f"- **Rows**: {len(records)}",
        "- **Columns**: case_id, video_id, timestamp_seconds, frame_index, frame_path, candidate_phase, candidate_use, manual_label, manual_notes, roi_candidate, confirmed_by",
        "- **Note**: `manual_label`, `manual_notes`, `confirmed_by` columns are EMPTY — must be filled by Codex / user",
        "",
        "## 6. Candidate Time Segmentation",
        "",
        "> **CRITICAL**: ALL segments below are CANDIDATES ONLY. They are NOT final labels.",
        "> Codex / user MUST confirm or revise before any experiment uses them.",
        "",
        f"### Candidate Bank Normal (候选，需 Codex / 用户确认)",
        "",
    ])

    # Heuristic: middle section 120-600s as bank normal (avoid startup/ending)
    if duration > 600:
        lines.append(f"- **t=120s – t=600s** (duration: 480s) — 可能为稳定运行期")
        lines.append(f"- **Reason**: avoids startup (0-120s) and late-stage (last 120s); middle of recording")
    else:
        lines.append(f"- **t=30s – t={duration-30:.0f}s** (duration: {duration-60:.0f}s) — narrow window, full middle section")
        lines.append(f"- **Reason**: video is short ({duration:.0f}s), using middle section excluding edges")

    lines.extend([
        "",
        f"### Candidate Calibration Normal (候选，需 Codex / 用户确认)",
        "",
        f"- **t=0s – t=120s** — 视频开头 2 分钟，包含设备初始状态",
        f"- **Purpose**: 可用于验证 memory bank 对初始帧的泛化能力",
        f"- **Risk**: 开头可能包含装配/设置操作，不一定是'纯正常'",
        "",
        f"### Candidate Test Stream (候选，需 Codex / 用户确认)",
        "",
        f"- **t=600s – t={duration:.0f}s** — 视频后半段",
        f"- **Purpose**: 可用于检测 memory bank 对时间远端帧的稳定性",
        f"- **Risk**: 如果后半段状态与前半段一致，可能不存在真正的'异常'，检测将退化为正常vs正常",
        "",
        "### Key Timestamps for Manual Review (候选，需 Codex / 用户确认)",
        "",
        "| Timestamp | Frame Index | Reason |",
        "|-----------|-------------|--------|",
        "| t=0.0s | 0 | Video start — initial equipment state |",
        f"| t=30.0s | ~900 | After startup — transition to stable |",
        f"| t=120.0s | ~3594 | End of startup window |",
        f"| t=300.0s | ~8985 | Mid-point of candidate bank normal |",
        f"| t=600.0s | ~17970 | Start of candidate test stream |",
    ])

    # Add every 60s as review points
    t = 60.0
    while t <= duration:
        frame_est = int(t * 29.95)
        lines.append(f"| t={t:.1f}s | ~{frame_est} | Routine checkpoint |")
        t += 120.0  # Every 2 minutes

    lines.extend([
        "",
        "## 7. Questions for Codex / User Confirmation",
        "",
        "1. **ROI selection**: Which ROI (1/2/3) is most appropriate for PatchCore memory bank? ROI 1 (dialyzer only) is narrowest and cleanest; ROI 3 (wide workstation) captures more context but adds noise.",
        "2. **Normal definition**: Is the entire video 'normal operation'? Or are there specific segments with known defects?",
        "3. **Bank Normal window**: Is t=120-600s truly representative of normal operation? Should the window be narrower/wider?",
        "4. **Test stream setup**: If the entire video is normal, what constitutes a 'test' stream — just a held-out normal segment? Or do you plan to inject synthetic anomalies?",
        "5. **Temporal leakage**: Current candidate split respects temporal order (early → bank/calibration → test). Is this acceptable, or would you prefer random frame sampling?",
        "6. **Auxiliary PNG**: Does the Case 4 screenshot (`屏幕截图 2026-04-27 092740.png`) show the same workstation? Does it provide ROI or label hints?",
        "7. **Frame-level vs clip-level**: Should anomaly detection operate on individual frames or temporal clips (e.g., 5-frame windows)?",
        "8. **Label granularity**: Is binary (normal/abnormal) sufficient, or do you need defect-type labels (misalignment, missing module, contamination)?",
        "",
        "## 8. Unresolved Issues",
        "",
        "- Case 4 DOCX (案例4 has no DOCX — only 案例2 has one) — no supplementary documentation found for case 4",
        "- No external ground-truth labels found for this video",
        "- The video appears to show a stable automated station; whether it contains any real anomalies is unknown",
        '- Dialyzer labels (F16, WAZOER, 同谱) suggest this is medical device production; domain-specific anomalies need expert definition',
        "",
        "## 9. Code Modifications",
        "",
        "- **Created**: `src/week2_prepare_review_pack.py` — review pack preparation script",
        "- **NOT modified**: `src/run_padim.py`, `src/run_patchcore.py`, `src/run_spade.py`, `src/run_anomalydino.py`",
        "",
        "## 10. Commands Executed",
        "",
        "```powershell",
        "conda activate braincv-ad-py310",
        "cd E:\\leinaozuoye",
        "python src/week2_prepare_review_pack.py",
        "```",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")


def main() -> int:
    print("=== Step 1: Extract key frames ===")
    records = extract_key_frames()

    print("\n=== Step 2: Generate ROI images ===")
    generate_roi_images(records)

    print("\n=== Step 3: Extract case 4 auxiliary PNG ===")
    png_path = extract_case4_png()

    print("\n=== Step 4: Generate label template ===")
    generate_label_template(records)

    print("\n=== Step 5: Write review frames manifest ===")
    write_review_frames_csv(records)

    print("\n=== Step 6: Write report ===")
    write_report(records, png_path, list(ROI_CANDIDATES.keys()))

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
