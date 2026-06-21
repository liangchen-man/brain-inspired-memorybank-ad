#!/usr/bin/env python3
"""Regenerate the case4 manual label template CSV with verified anomaly labels."""

import csv
from pathlib import Path

REVIEW_DIR = Path(r"E:\leinaozuoye\data\workstation\review_frames\20241101_161258")
ANOMALY_DIR = Path(r"E:\leinaozuoye\data\workstation\anomaly_frames\case4")
OUT = Path(r"E:\leinaozuoye\data\workstation\manifests\week2_case4_manual_label_template.csv")

# --------------- helper ---------------
def classify(t: float):
    """Return (candidate_phase, candidate_use, manual_label, manual_notes) for a timestamp."""
    if t < 120:
        phase = "possible_startup" if t < 40 else "possible_stable"
        return (phase, "candidate_calibration", "normal",
                "Calibration window (0-120s). Verified via sampled frames: all caps present.")
    if 120 <= t < 260:
        return ("possible_stable", "candidate_bank_normal", "normal",
                "Bank Normal window (120-260s). t=180s verified by qwen3-vl-plus: all 5 dialyzers have all caps present.")
    # Test stream starts at 260s
    if 260 <= t < 272:
        return ("possible_stable", "candidate_test", "",
                "Test Stream start. Before first known anomaly.")
    if 272 <= t < 949:
        if t == 270.0 or t == 272.0:
            return ("candidate_anomaly", "candidate_test", "anomaly",
                    "Anomaly1 (~272s). T1 side cap missing per auxiliary PNG. qwen3-vl-plus: NOT VISIBLE in this camera angle (mechanical arm occlusion). Needs human confirmation.")
        return ("possible_stable", "candidate_test", "",
                "Between Anomaly1 and Anomaly2. Presumed normal operation.")
    if 949 <= t < 1047:
        if t in (930.0, 949.0, 960.0):
            return ("candidate_anomaly", "candidate_test", "anomaly",
                    "Anomaly2 (~949s). T1 arterial side cap MISSING + T5 venous side cap MISSING. Both CONFIRMED VISIBLE by qwen3-vl-plus. BEST verification point.")
        return ("possible_stable", "candidate_test", "",
                "Near Anomaly2 region.")
    if 1047 <= t < 1072:
        if t in (1041.2, 1047.0, 1050.0, 1051.2):
            return ("candidate_anomaly", "candidate_test", "anomaly",
                    "Anomaly3 (~1047s). T2 right side cap MISSING + T5 right side cap MISSING. Both CONFIRMED VISIBLE by qwen3-vl-plus.")
        return ("possible_stable", "candidate_test", "",
                "Near Anomaly3 region.")
    # t >= 1072
    if t in (1071.2, 1072.0):
        return ("candidate_anomaly", "candidate_test", "anomaly",
                "Anomaly4 (~1072s). T5 right side cap MISSING. CONFIRMED VISIBLE by qwen3-vl-plus. T5 has been missing this cap since t=949s (persistent anomaly).")
    if t > 1080:
        return ("possible_ending", "candidate_test", "",
                "Video end. Final frames.")
    return ("possible_stable", "candidate_test", "", "Routine checkpoint.")

# --------------- collect existing review frames ---------------
existing = {}
for png in sorted(REVIEW_DIR.glob("case4_t*.png")):
    stem = png.stem  # e.g. case4_t00000.0s
    t_str = stem.replace("case4_t", "").replace("s", "")
    try:
        t = float(t_str)
    except ValueError:
        continue
    existing[t] = png

# --------------- add anomaly exact frames ---------------
anomaly_frames = {
    180.0: ANOMALY_DIR / "t0180.0s_normal_baseline.png",
    272.0: ANOMALY_DIR / "t0272.0s_anomaly1.png",
    949.0: ANOMALY_DIR / "t0949.0s_anomaly2.png",
    1047.0: ANOMALY_DIR / "t1047.0s_anomaly3.png",
    1072.0: ANOMALY_DIR / "t1072.0s_anomaly4.png",
}

# --------------- build rows ---------------
rows = []
for t in sorted(set(list(existing.keys()) + list(anomaly_frames.keys()))):
    path = anomaly_frames.get(t) or existing.get(t)
    if path is None:
        continue
    phase, use, label, notes = classify(t)
    rows.append({
        "case_id": "case4",
        "video_id": "20241101_161258",
        "timestamp_seconds": f"{t:.1f}",
        "frame_index": "",
        "frame_path": str(path),
        "candidate_phase": phase,
        "candidate_use": use,
        "manual_label": label,
        "manual_notes": notes,
        "roi_candidate": "待确认",
        "confirmed_by": "",
    })

# --------------- write ---------------
FIELD_NAMES = [
    "case_id", "video_id", "timestamp_seconds", "frame_index",
    "frame_path", "candidate_phase", "candidate_use",
    "manual_label", "manual_notes", "roi_candidate", "confirmed_by",
]

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    writer.writeheader()
    writer.writerows(rows)

# --------------- stats ---------------
n_anomaly = sum(1 for r in rows if r["manual_label"] == "anomaly")
n_normal = sum(1 for r in rows if r["manual_label"] == "normal")
n_empty = sum(1 for r in rows if r["manual_label"] == "")
print(f"CSV written: {OUT}")
print(f"Total rows: {len(rows)}")
print(f"  anomaly: {n_anomaly}")
print(f"  normal:  {n_normal}")
print(f"  empty:   {n_empty}")
print(f"Anomaly timestamps: {[r['timestamp_seconds'] for r in rows if r['manual_label'] == 'anomaly']}")
