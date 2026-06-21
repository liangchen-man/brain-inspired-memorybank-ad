#!/usr/bin/env python3
"""Week 2 PatchCore video migration runner for Case 4 dialyzer workstation.

Reads frames from video via OpenCV, crops to ROI, extracts patch features with
ResNet18 (reuses Week 1 backbone), builds a memory bank from Bank Normal frames,
and scores every frame in the Test Stream. Produces a per-frame anomaly score
timeline, heatmaps for top anomaly frames, and summary statistics.

Design constraints:
- Reuses ResNet18Layer2 from run_patchcore.py (Week 1 consistency).
- ROI 1 (480,162,1190,950) is the default; override with --roi.
- Bank Normal: t=120-260s (default), Test Stream: t=260-end.
- Sampling: ~0.5 FPS by default (every Nth frame).
- All scores saved as CSV; human labels read from the manual label template.

GPU memory: ~32 MB peak (ResNet18), runs comfortably on RTX 3050 4GB.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

# ---- Reuse Week 1 ResNet18 backbone ----
class ResNet18Layer2(nn.Module):
    def __init__(self, weights: str = "imagenet") -> None:
        super().__init__()
        if weights == "imagenet":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif weights == "none":
            backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported weights mode: {weights}")
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ---- Image transform (matches Week 1) ----
def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---- Video frame iterator ----
def iter_video_frames(
    video_path: Path,
    roi: tuple[int, int, int, int],
    fps: float,
    sample_interval: float,
    t_start: float,
    t_end: float,
    image_size: int,
) -> list[dict]:
    """Extract frames from video at `sample_interval` seconds within [t_start, t_end).
    Crop each frame to `roi` = (x1, y1, x2, y2), resize to image_size x image_size,
    and return a list of dicts with 'timestamp', 'frame_idx', 'tensor', 'roi_array'.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    t_end = min(t_end, duration)

    transform = make_transform(image_size)
    x1, y1, x2, y2 = roi

    records: list[dict] = []
    t = t_start
    while t < t_end:
        frame_idx = int(t * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            t += sample_interval
            continue

        # Crop ROI
        h, w = frame_bgr.shape[:2]
        rx1 = max(0, min(x1, w - 1))
        ry1 = max(0, min(y1, h - 1))
        rx2 = max(0, min(x2, w))
        ry2 = max(0, min(y2, h))
        if rx2 <= rx1 or ry2 <= ry1:
            t += sample_interval
            continue

        crop = frame_bgr[ry1:ry2, rx1:rx2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(crop_rgb)
        tensor = transform(pil_img)

        records.append({
            "timestamp": round(t, 2),
            "frame_idx": frame_idx,
            "tensor": tensor,
            "roi_array": crop_rgb,  # for heatmap overlay
        })
        t += sample_interval

    cap.release()
    return records


# ---- Patch feature extraction ----
def extract_patch_features(
    model: nn.Module,
    frames: list[dict],
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], int, int]:
    """Extract patch features from all frames. Returns list of feature tensors,
    channel count, and feature spatial size (h, w)."""
    model.eval()
    all_feats: list[torch.Tensor] = []
    ch = None
    f_hw = None

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch = torch.stack([f["tensor"] for f in batch_frames]).to(device)
            feats = model(batch).detach().cpu()
            if ch is None:
                ch = feats.shape[1]
                f_hw = (feats.shape[2], feats.shape[3])
            # per-frame: (c, h, w) -> (h*w, c)
            for j in range(feats.shape[0]):
                single = feats[j]  # (c, h, w)
                flat = single.permute(1, 2, 0).reshape(-1, ch)
                all_feats.append(F.normalize(flat, dim=1))

    if ch is None or f_hw is None:
        raise RuntimeError("No features extracted")
    return all_feats, ch, f_hw


# ---- Coreset (same as Week 1) ----
def build_memory_bank(
    patch_list: list[torch.Tensor],
    coreset_size: int,
    max_candidates: int,
    method: str,
    seed: int,
    chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    bank = torch.cat(patch_list, dim=0).contiguous()
    n = bank.shape[0]

    # Candidate subset
    if n > max_candidates:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=gen)[:max_candidates]
        candidates = bank[idx].contiguous()
    else:
        candidates = bank

    target = min(coreset_size, candidates.shape[0])
    if method == "random":
        if candidates.shape[0] <= target:
            coreset = candidates
        else:
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(candidates.shape[0], generator=gen)[:target]
            coreset = candidates[idx].contiguous()
    elif method == "greedy":
        coreset = _greedy_coreset(candidates, target, seed, chunk_size)
    else:
        raise ValueError(f"Unsupported coreset method: {method}")

    info = {
        "raw_patch_count": n,
        "candidate_patch_count": int(candidates.shape[0]),
        "coreset_patch_count": int(coreset.shape[0]),
        "coreset_method": method,
        "feature_dim": int(bank.shape[1]),
    }
    return coreset, info


def _greedy_coreset(
    bank: torch.Tensor, target: int, seed: int, chunk_size: int
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    first = int(torch.randint(0, bank.shape[0], (1,), generator=gen).item())
    selected = [first]
    min_dist = torch.full((bank.shape[0],), float("inf"))
    for _ in range(1, target):
        latest = bank[selected[-1] : selected[-1] + 1]
        for s in range(0, bank.shape[0], chunk_size):
            e = min(s + chunk_size, bank.shape[0])
            d = torch.cdist(bank[s:e], latest).squeeze(1)
            min_dist[s:e] = torch.minimum(min_dist[s:e], d)
        selected.append(int(torch.argmax(min_dist).item()))
    return bank[torch.tensor(selected)].contiguous()


# ---- Scoring ----
def score_frames(
    model: nn.Module,
    frame_feats: list[torch.Tensor],
    memory_bank: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> list[np.ndarray]:
    """Score every frame: per-patch NN distance to memory bank.
    Returns list of (h, w) score maps (one per frame)."""
    score_maps: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for patches in frame_feats:
            patches_dev = patches.to(device)
            mem_dev = memory_bank.to(device)
            # chunked cdist
            all_min: list[torch.Tensor] = []
            for s in range(0, patches_dev.shape[0], chunk_size):
                e = min(s + chunk_size, patches_dev.shape[0])
                d = torch.cdist(patches_dev[s:e], mem_dev)
                all_min.append(d.min(dim=1).values.cpu())
            dists = torch.cat(all_min, dim=0)
            score_maps.append(dists.numpy())
    return score_maps


def image_score(score_map: np.ndarray) -> float:
    return float(score_map.max())


# ---- Visuals ----
def save_heatmap_overlay(
    roi_array: np.ndarray,
    score_map: np.ndarray,
    out_path: Path,
) -> None:
    """Upsample score_map to roi_array size, overlay heatmap on original crop."""
    h, w = roi_array.shape[:2]
    sh, sw = score_map.shape
    if sh != h or sw != w:
        up = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        up = score_map
    # normalize
    lo, hi = float(up.min()), float(up.max())
    if math.isclose(hi, lo):
        norm = np.zeros_like(up, dtype=np.uint8)
    else:
        norm = ((up - lo) / (hi - lo) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(roi_array, 0.55, heat, 0.45, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


# ---- Load manual labels ----
def load_labels(label_csv: Path) -> dict[float, str]:
    """Return {timestamp_seconds: manual_label} for rows where manual_label is non-empty."""
    labels: dict[float, str] = {}
    if not label_csv.is_file():
        return labels
    with open(label_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("manual_label", "").strip()
            if lbl:
                try:
                    t = float(row["timestamp_seconds"])
                except ValueError:
                    continue
                labels[t] = lbl
    return labels


# ---- Main ----
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 2 PatchCore video migration for Case 4")
    p.add_argument("--video", type=Path,
                   default=Path("data/workstation/raw/20241101_161258.mp4"))
    p.add_argument("--output", type=Path, default=None,
                   help="Output dir (default: runs/week2_patchcore_case4)")
    p.add_argument("--roi", type=int, nargs=4, default=(480, 162, 1190, 950),
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="ROI pixel coords (default: ROI 1 = 480 162 1190 950)")
    p.add_argument("--bank-start", type=float, default=120.0)
    p.add_argument("--bank-end", type=float, default=260.0)
    p.add_argument("--test-start", type=float, default=260.0)
    p.add_argument("--test-end", type=float, default=0.0,
                   help="0 = use full video duration")
    p.add_argument("--sample-interval", type=float, default=2.0,
                   help="Seconds between sampled frames (default 2.0 = 0.5 FPS)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--coreset-size", type=int, default=1000)
    p.add_argument("--max-candidates", type=int, default=20000)
    p.add_argument("--coreset-method", choices=["random", "greedy"], default="random")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--nn-chunk-size", type=int, default=2048)
    p.add_argument("--max-heatmaps", type=int, default=30)
    p.add_argument("--label-csv", type=Path,
                   default=Path("data/workstation/manifests/week2_case4_manual_label_template.csv"))
    p.add_argument("--top-anomaly-n", type=int, default=30,
                   help="Number of top-scoring frames to save heatmaps for")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()
    command = "python " + " ".join(sys.argv)

    output_dir = args.output or Path("runs") / "week2_patchcore_case4"
    heatmap_dir = output_dir / "heatmaps"
    for d in [output_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    scores_csv_path = output_dir / "frame_scores.csv"
    timeline_path = output_dir / "anomaly_timeline.csv"
    memory_path = output_dir / "memory_bank.pt"
    run_log_path = output_dir / "run_log.md"

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    result: dict = {
        "algorithm": "PatchCore-teaching-video",
        "video": str(args.video),
        "roi": list(args.roi),
        "status": "failed",
        "command": command,
    }
    log_lines: list[str] = [
        "# Week 2 PatchCore Video Run Log",
        "",
        f"- Video: `{args.video}`",
        f"- ROI: {args.roi}",
        f"- Bank Normal: t={args.bank_start}-{args.bank_end}s",
        f"- Test Stream: t={args.test_start}-{args.test_end if args.test_end > 0 else 'end'}s",
        f"- Sample interval: {args.sample_interval}s (~{1.0/args.sample_interval:.2f} FPS)",
        f"- Command: `{command}`",
        "",
    ]

    try:
        # ---- 1. Load labels ----
        labels = load_labels(args.label_csv)
        print(f"Loaded {len(labels)} manual labels from {args.label_csv}")
        log_lines.append(f"- Manual labels loaded: {len(labels)}")

        # ---- 2. Extract Bank Normal frames (also peek duration) ----
        cap_temp = cv2.VideoCapture(str(args.video))
        video_fps = cap_temp.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        cap_temp.release()

        test_end = args.test_end if args.test_end > 0 else duration
        print(f"Video: {args.video}")
        print(f"  Duration: {duration:.1f}s, FPS: {video_fps:.2f}, Total frames: {total_frames}")
        print(f"  Bank Normal: t={args.bank_start}-{args.bank_end}s")
        print(f"  Test Stream: t={args.test_start}-{test_end:.1f}s")
        print(f"  Sample interval: {args.sample_interval}s")

        model = ResNet18Layer2(args.weights).to(device).eval()

        # Build memory bank
        bank_frames = iter_video_frames(
            args.video, args.roi, video_fps, args.sample_interval,
            args.bank_start, args.bank_end, args.image_size,
        )
        print(f"Bank Normal frames: {len(bank_frames)}")
        if len(bank_frames) == 0:
            raise RuntimeError("No Bank Normal frames extracted — check ROI and time window")

        bank_feats, feat_channels, feat_hw = extract_patch_features(
            model, bank_frames, device, args.batch_size,
        )

        memory_bank, mem_info = build_memory_bank(
            bank_feats, args.coreset_size, args.max_candidates,
            args.coreset_method, args.seed, args.nn_chunk_size,
        )
        torch.save({
            "memory_bank": memory_bank,
            "feature_channels": feat_channels,
            "feature_hw": feat_hw,
            "memory_info": mem_info,
            "roi": args.roi,
            "image_size": args.image_size,
        }, memory_path)
        print(f"Memory bank: {mem_info['coreset_patch_count']} patches (raw: {mem_info['raw_patch_count']})")

        # ---- 3. Extract and score Test Stream frames ----
        test_frames = iter_video_frames(
            args.video, args.roi, video_fps, args.sample_interval,
            args.test_start, test_end, args.image_size,
        )
        print(f"Test Stream frames: {len(test_frames)}")

        test_feats, _, _ = extract_patch_features(
            model, test_frames, device, args.batch_size,
        )

        score_maps = score_frames(model, test_feats, memory_bank, device, args.nn_chunk_size)
        print(f"Scored {len(score_maps)} test frames")

        # ---- 4. Aggregate results ----
        frame_results: list[dict] = []
        for i, (frec, smap) in enumerate(zip(test_frames, score_maps)):
            smap_blur = cv2.GaussianBlur(smap, (5, 5), 0)
            img_score = float(smap_blur.max())
            t = frec["timestamp"]
            nearest_label = None
            nearest_label_t = None
            min_dist_label = 999.0
            for lt in labels:
                d = abs(t - lt)
                if d < min_dist_label:
                    min_dist_label = d
                    nearest_label = labels[lt]
                    nearest_label_t = lt
            frame_results.append({
                "timestamp": t,
                "frame_idx": frec["frame_idx"],
                "anomaly_score": img_score,
                "score_mean": float(smap_blur.mean()),
                "score_std": float(smap_blur.std()),
                "nearest_label_t": round(nearest_label_t, 1) if nearest_label_t else None,
                "manual_label": nearest_label if min_dist_label < args.sample_interval * 2 else None,
            })

        # ---- 5. Build anomaly timeline (all frames, sorted by score desc) ----
        timeline = sorted(frame_results, key=lambda r: r["anomaly_score"], reverse=True)

        # ---- 6. Save heatmaps for top anomaly frames ----
        saved_heatmaps = 0
        for entry in timeline[:args.top_anomaly_n]:
            if saved_heatmaps >= args.max_heatmaps:
                break
            idx = next(i for i, r in enumerate(frame_results) if r["timestamp"] == entry["timestamp"])
            smap = score_maps[idx]
            smap_blur = cv2.GaussianBlur(smap, (5, 5), 0)
            roi_arr = test_frames[idx]["roi_array"]
            t_str = f"{entry['timestamp']:.1f}s".replace(".", "p")
            label_str = entry.get("manual_label") or "unknown"
            out_name = f"t{t_str}_score{entry['anomaly_score']:.4f}_{label_str}.png"
            save_heatmap_overlay(roi_arr, smap_blur, heatmap_dir / out_name)
            saved_heatmaps += 1

        # ---- 7. Compute hit-rate: top-N overlap with anomaly labels ----
        anomaly_timestamps = {t for t, lbl in labels.items() if lbl == "anomaly"}
        normal_timestamps = {t for t, lbl in labels.items() if lbl == "normal"}
        hit_details: list[dict] = []
        for k in [5, 10, 20, 30]:
            top_k = timeline[:k]
            top_ts = {r["timestamp"] for r in top_k}
            # Find anomaly labels within sample_interval of top-k frames
            hits = 0
            for r in top_k:
                t = r["timestamp"]
                for at in anomaly_timestamps:
                    if abs(t - at) < args.sample_interval * 1.5:
                        hits += 1
                        break
            hit_details.append({
                "top_k": k,
                "hits": hits,
                "total_anomaly_labels": len(anomaly_timestamps),
                "hit_rate": round(hits / max(len(anomaly_timestamps), 1), 3),
            })
        print(f"Top-K hit rate vs anomaly labels: {hit_details}")

        # ---- 8. Write outputs ----
        # frame_scores.csv
        with open(scores_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "frame_idx", "anomaly_score", "score_mean",
                "score_std", "nearest_label_t", "manual_label",
            ])
            writer.writeheader()
            writer.writerows(sorted(frame_results, key=lambda r: r["timestamp"]))

        # anomaly_timeline.csv (ranked by score)
        with open(timeline_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "rank", "timestamp", "frame_idx", "anomaly_score",
                "score_mean", "score_std", "manual_label",
            ])
            writer.writeheader()
            for rank, entry in enumerate(timeline, 1):
                writer.writerow({
                    "rank": rank,
                    "timestamp": entry["timestamp"],
                    "frame_idx": entry["frame_idx"],
                    "anomaly_score": entry["anomaly_score"],
                    "score_mean": entry["score_mean"],
                    "score_std": entry["score_std"],
                    "manual_label": entry.get("manual_label") or "",
                })

        # results.json
        runtime = round(time.perf_counter() - start_time, 3)
        result.update({
            "status": "success",
            "runtime_seconds": runtime,
            "max_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 3)
                if use_cuda else None,
            "bank_normal_frames": len(bank_frames),
            "test_frames": len(test_frames),
            "bank_normal_window": [args.bank_start, args.bank_end],
            "test_window": [args.test_start, round(test_end, 1)],
            "sample_interval": args.sample_interval,
            "memory_bank_info": mem_info,
            "memory_bank_path": str(memory_path),
            "memory_bank_bytes": memory_path.stat().st_size,
            "top_k_anomaly_hits": hit_details,
            "heatmaps_saved": saved_heatmaps,
            "frame_scores_csv": str(scores_csv_path),
            "anomaly_timeline_csv": str(timeline_path),
            "output_dir": str(output_dir),
            "error": None,
        })
        results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        log_lines.extend([
            f"- Status: **success**",
            f"- Device: {device}",
            f"- Bank Normal frames: {len(bank_frames)}",
            f"- Test Stream frames: {len(test_frames)}",
            f"- Memory bank size: {mem_info['coreset_patch_count']} patches",
            f"- Runtime: {runtime}s",
            f"- Max GPU memory: {result['max_gpu_mem_mb']} MB",
            f"- Heatmaps saved: {saved_heatmaps}",
            f"- Top-10 anomaly hit rate: {hit_details[1]['hit_rate'] if len(hit_details) > 1 else 'N/A'}",
        ])
        run_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        runtime = round(time.perf_counter() - start_time, 3)
        result["runtime_seconds"] = runtime
        result["error"] = f"{type(exc).__name__}: {exc}"
        results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log_lines.append(f"- Status: **failed**")
        log_lines.append(f"- Error: {result['error']}")
        run_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
