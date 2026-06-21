#!/usr/bin/env python3
"""Week2 Case4 Visibility Upper-Bound Baseline — v3 (sub-ROI averaging).

用更大侧帽子区域 + Bank Normal 帧平均做参考，直接对比每个异常帧。
不依赖精确单点坐标——用 40×50 的子 ROI 覆盖侧帽区域 + 边距，
Bank Normal 帧取中值平均抑制噪声，然后对每个 query 帧做 L1/L2/SSIM/边缘/颜色对比。

还做：3-5 帧时序聚合 + 差异图可视化。

输出：
  results/week2/case4_visibility_baseline.csv
  runs/week2_visibility_baseline/patches/*.png
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "workstation" / "raw" / "20241101_161258.mp4"

# 子 ROI 定义：每个透析器 RIGHT side 侧帽附近区域 (full-frame x1,y1,x2,y2)
# T1-T3 坐标由 qwen3-vl-plus 在校验帧上重新标定 (右侧帽)
# T4-T5 坐标经确认正确 (已标记在 right side)
_HW, _HH = 20, 25
SIDE_CAP_SUB_ROIS = {
    "T1": (280 - _HW, 545 - _HH, 280 + _HW, 545 + _HH),  # RIGHT cap for T1
    "T2": (440 - _HW, 545 - _HH, 440 + _HW, 545 + _HH),  # RIGHT cap for T2
    "T3": (600 - _HW, 545 - _HH, 600 + _HW, 545 + _HH),  # RIGHT cap for T3
    "T4": (650 - _HW, 610 - _HH, 650 + _HW, 610 + _HH),  # RIGHT cap (confirmed)
    "T5": (805 - _HW, 615 - _HH, 805 + _HW, 615 + _HH),  # RIGHT cap (confirmed)
}

BANK_START = 120.0
BANK_END = 260.0
NORMAL_BASELINE_TS = 180.0
N_BANK_FRAMES = 20

ANOMALY_TIMESTAMPS = [
    ("t=272s_anomaly", 272.0, "anomaly"),
    ("t=949s_anomaly", 949.0, "anomaly"),
    ("t=1047s_anomaly", 1047.0, "anomaly"),
    ("t=1072s_anomaly", 1072.0, "anomaly"),
]
EXTRA_NORMAL_TS = [
    ("t=300s", 300.0, "extra"),
    ("t=500s", 500.0, "extra"),
    ("t=700s", 700.0, "extra"),
    ("t=900s", 900.0, "extra"),
    ("t=1050s", 1050.0, "extra"),
]
TIMESTAMP_WINDOW_RADIUS = 2  # frames to aggregate around each timestamp


def get_frame_at(video, target_sec):
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    video.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
    ok, frame = video.read()
    if not ok:
        return None, target_sec
    return frame, int(target_sec * fps) / fps


def extract_sub_roi(gray, dialyzer_id):
    x1, y1, x2, y2 = SIDE_CAP_SUB_ROIS[dialyzer_id]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    return gray[y1:y2, x1:x2]


def compute_metrics(a, b):
    """L1, L2, SSIM, edge L2, hist correlation between two same-size grayscale patches."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    af, bf = a.astype(np.float32), b.astype(np.float32)
    diff = af - bf
    l1 = float(np.abs(diff).mean())
    l2 = float(np.sqrt((diff ** 2).mean()))
    s = ssim(a, b, data_range=255)
    sa = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=3)
    sb = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
    el2 = float(np.sqrt(((sa - sb) ** 2).mean()))
    ha = cv2.calcHist([a], [0], None, [32], [0, 256])
    hb = cv2.calcHist([b], [0], None, [32], [0, 256])
    hc = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return {"l1": round(l1, 3), "l2": round(l2, 3), "ssim": round(s, 4),
            "edge_l2": round(el2, 3), "hist_corr": round(hc, 4)}


def main():
    parser = argparse.ArgumentParser(description="Case4 Visibility Baseline v3")
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--outdir", default=str(ROOT / "runs" / "week2_visibility_baseline"))
    parser.add_argument("--n-bank", type=int, default=N_BANK_FRAMES)
    parser.add_argument("--window", type=int, default=TIMESTAMP_WINDOW_RADIUS)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    patch_dir, diff_dir = out_dir / "patches", out_dir / "diffs"
    for d in [out_dir, patch_dir, diff_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Sub-ROIs: { {k: v for k, v in SIDE_CAP_SUB_ROIS.items()} }")
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"Video: {args.video}, FPS: {fps:.2f}")

    # ---- Step 1: Build bank normal median reference ----
    print(f"\n[1] Building median reference from {args.n_bank} bank frames (t={BANK_START}-{BANK_END}s)...")
    bank_frames = {}
    for dialyzer in SIDE_CAP_SUB_ROIS:
        bank_frames[dialyzer] = []
    for k in range(args.n_bank):
        t = BANK_START + k * (BANK_END - BANK_START) / max(1, args.n_bank - 1)
        frame, _ = get_frame_at(video, t)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for dialyzer in SIDE_CAP_SUB_ROIS:
            roi = extract_sub_roi(gray, dialyzer)
            bank_frames[dialyzer].append(roi.astype(np.float32))

    references = {}
    for dialyzer, patches in bank_frames.items():
        stack = np.stack(patches, axis=0)
        references[dialyzer] = np.median(stack, axis=0).astype(np.uint8)
        print(f"  {dialyzer}: {len(patches)} patches, shape={stack.shape[1:]}")

    # Save reference patches
    for dialyzer in SIDE_CAP_SUB_ROIS:
        cv2.imwrite(str(patch_dir / f"ref_median_{dialyzer}.png"), references[dialyzer])

    # ---- Step 2: Score all timestamps with temporal aggregation ----
    print(f"\n[2] Scoring timestamps (window ±{args.window} frames)...")
    all_ts = [("t=180s_normal_baseline", NORMAL_BASELINE_TS, "normal")] + \
             ANOMALY_TIMESTAMPS + EXTRA_NORMAL_TS

    rows = []
    for label, ts, category in sorted(all_ts, key=lambda x: x[1]):
        # Collect window frames
        window_patches = {dialyzer: [] for dialyzer in SIDE_CAP_SUB_ROIS}
        n_collected = 0
        for offset in range(-args.window, args.window + 1):
            t = ts + offset / fps
            frame, _ = get_frame_at(video, t)
            if frame is None:
                continue
            n_collected += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for dialyzer in SIDE_CAP_SUB_ROIS:
                roi = extract_sub_roi(gray, dialyzer)
                window_patches[dialyzer].append(roi)

        if n_collected == 0:
            print(f"  SKIP {label}: zero frames")
            continue

        for dialyzer in SIDE_CAP_SUB_ROIS:
            patches = window_patches[dialyzer]
            # Compute per-frame metrics, then average across window
            frame_metrics = []
            for patch in patches:
                m = compute_metrics(patch, references[dialyzer])
                frame_metrics.append(m)
            # Average across window
            agg = {}
            for key in ["l1", "l2", "ssim", "edge_l2", "hist_corr"]:
                vals = [fm[key] for fm in frame_metrics]
                agg[f"{key}_mean"] = round(float(np.mean(vals)), 3)
                agg[f"{key}_std"] = round(float(np.std(vals)), 4)
            agg["n_frames"] = len(frame_metrics)
            rows.append({
                "timestamp": round(ts, 1), "label": label, "category": category,
                "dialyzer": dialyzer, **agg,
            })

    # ---- Step 3: Save reference patch PNGs + Save query patch PNGs for key frames ----
    print(f"\n[3] Saving debug images...")
    for label, ts, category in all_ts:
        frame, _ = get_frame_at(video, ts)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for dialyzer in SIDE_CAP_SUB_ROIS:
            if category in ("anomaly", "normal"):
                roi = extract_sub_roi(gray, dialyzer)
                cv2.imwrite(str(patch_dir / f"{label}_{dialyzer}.png"), roi)
                # Difference map (amplified 5x)
                ref = references[dialyzer]
                if roi.shape == ref.shape:
                    d = cv2.absdiff(roi.astype(np.float32), ref.astype(np.float32))
                    dv = np.clip(d * 4, 0, 255).astype(np.uint8)
                    dh = cv2.applyColorMap(dv, cv2.COLORMAP_HOT)
                    ref_c = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
                    qry_c = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    panel = np.hstack([ref_c, qry_c, dh])
                    cv2.imwrite(str(diff_dir / f"{label}_{dialyzer}_diff.png"), panel)

    # ---- Step 4: Save CSV ----
    csv_path = ROOT / "results" / "week2" / "case4_visibility_baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "label", "category", "dialyzer",
                  "l1_mean", "l1_std", "l2_mean", "l2_std",
                  "ssim_mean", "ssim_std", "edge_l2_mean", "edge_l2_std",
                  "hist_corr_mean", "hist_corr_std", "n_frames"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path} ({len(rows)} rows)")

    # ---- Step 5: Analysis ----
    print(f"\n{'='*70}")
    print("DETECTABILITY ANALYSIS")
    print("=" * 70)

    for dialyzer in ["T1", "T2", "T3", "T4", "T5"]:
        d_rows = [r for r in rows if r["dialyzer"] == dialyzer]
        # Extra frames (non-bank normal checkpoints) to establish normal variation baseline
        extra_l2 = [r["l2_mean"] for r in d_rows if r["category"] == "extra"]
        normal_l2 = [r["l2_mean"] for r in d_rows if r["category"] == "normal"]
        anomaly_rows = [r for r in d_rows if r["category"] == "anomaly"]

        baseline = extra_l2 + normal_l2
        if not baseline:
            continue
        mu = np.mean(baseline)
        sigma = np.std(baseline) if len(baseline) > 1 else 1.0

        print(f"\n  {dialyzer}: normal mu={mu:.2f} sigma={sigma:.2f} (from {len(baseline)} non-anomaly frames)")
        for r in sorted(anomaly_rows, key=lambda r: -r["l2_mean"]):
            z = (r["l2_mean"] - mu) / sigma if sigma > 0 else 0
            if z > 3:
                tag = "*** DETECTABLE (z>3)"
            elif z > 1.5:
                tag = "**  MARGINAL (z>1.5)"
            else:
                tag = "*   NOT DETECTABLE"
            print(f"    {r['label']:30s} L2={r['l2_mean']:.2f}  SSIM={r['ssim_mean']:.4f}  z={z:+.1f}  {tag}")

    video.release()
    print("\nDone.")
    print(f"Output: {out_dir}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
