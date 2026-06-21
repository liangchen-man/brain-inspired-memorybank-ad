#!/usr/bin/env python3
"""Week 2 PatchCore video runner v2 — DINOv2 backbone + tight side-cap ROI.

Key changes from v1:
- --backbone dinov2|resnet18 (default dinov2). DINOv2 uses vit_small_patch14_dinov2
  via timm, extracting patch tokens (same as Week 1 AnomalyDINO runner).
- Tighter default ROI focused on the side-cap horizontal band (y=380-580) instead
  of the full dialyzer body. This makes the white side cap occupy more pixels.
- Heatmaps saved as PNG (lossless) to avoid JPEG stripe artifacts.
- All other CLI flags unchanged from v1.
"""

from __future__ import annotations

import argparse, csv, json, math, sys, time
from pathlib import Path

import cv2, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


# ═══════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ═══════════════════════════════════════════════════════════════════
# Backbones
# ═══════════════════════════════════════════════════════════════════

class ResNet18Layer2(nn.Module):
    """Week 1 ResNet18 backbone (unchanged)."""
    def __init__(self, weights: str = "imagenet") -> None:
        super().__init__()
        w = models.ResNet18_Weights.DEFAULT if weights == "imagenet" else None
        backbone = models.resnet18(weights=w)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # (B, C, H, W)


class DINOv2PatchTokens(nn.Module):
    """Week 1 DINOv2 ViT patch-token extractor via timm.

    DINOv2 models are pretrained at native resolution (518 for vit_small).
    We use img_size=518 to match pretrained weights, then resize input images
    to 518×518 in the transform. The feature_dim is applied via channel subsampling
    on the DINOv2 feature dimension (typically 384 for vit_small).
    """
    def __init__(self, model_name: str, image_size: int, feature_dim: int, seed: int) -> None:
        super().__init__()
        import timm
        # DINOv2 pretrained at 518; override image_size to match
        self.native_size = 518
        self.model = timm.create_model(model_name, pretrained=True, img_size=self.native_size, num_classes=0)
        self.feature_dim = feature_dim
        self.seed = seed
        self.channel_idx: torch.Tensor | None = None
        self.token_grid: tuple[int, int] | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(images)
        # Unwrap timm dict output
        if isinstance(feats, dict):
            tokens = None
            for key in ["x_norm_patchtokens", "x_patchtokens", "patch_tokens"]:
                v = feats.get(key)
                if v is not None:
                    tokens = v; break
            if tokens is None:
                tokens = feats.get("x_norm", feats.get("x"))
            feats = tokens
        if feats is None:
            raise RuntimeError("Could not extract patch tokens from timm output")

        b, n, c = feats.shape
        grid = int(math.sqrt(n))
        if grid * grid != n:
            # Drop CLS token if present
            feats = feats[:, 1:, :]; n = feats.shape[1]
            grid = int(math.sqrt(n))
            if grid * grid != n:
                raise RuntimeError(f"Token count {n} is not a square grid")

        if self.channel_idx is None:
            gen = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(c, generator=gen)[:self.feature_dim]
            self.channel_idx = perm
            self.token_grid = (grid, grid)

        reduced = feats[:, :, self.channel_idx.to(feats.device)]
        return F.normalize(reduced, dim=2)  # (B, N, D)


def build_backbone(name: str, image_size: int, feature_dim: int, seed: int) -> nn.Module:
    if name == "resnet18":
        return ResNet18Layer2("imagenet")
    if name == "dinov2":
        # DINOv2 is pretrained at 518; ignore the CLI image_size
        return DINOv2PatchTokens("vit_small_patch14_dinov2", 518, feature_dim, seed)
    raise ValueError(f"Unknown backbone: {name}")


# ═══════════════════════════════════════════════════════════════════
# Video frame iterator (same as v1)
# ═══════════════════════════════════════════════════════════════════

def iter_video_frames(
    video_path: Path, roi: tuple[int,int,int,int],
    video_fps: float, sample_interval: float,
    t_start: float, t_end: float, image_size: int,
) -> list[dict]:
    x1, y1, x2, y2 = roi
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(video_fps, 1)
    t_end = min(t_end, duration)

    transform = make_transform(image_size)
    records: list[dict] = []
    t = t_start
    while t < t_end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * video_fps))
        ret, frame = cap.read()
        if not ret:
            t += sample_interval; continue
        h, w = frame.shape[:2]
        rx1, ry1 = max(0, min(x1,w-1)), max(0, min(y1,h-1))
        rx2, ry2 = max(0, min(x2,w)),   max(0, min(y2,h))
        if rx2 <= rx1 or ry2 <= ry1:
            t += sample_interval; continue
        crop = frame[ry1:ry2, rx1:rx2]
        records.append({
            "timestamp": round(t, 2),
            "frame_idx": int(t * video_fps),
            "tensor": transform(pil_from_bgr(crop)),
            "roi_array": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        })
        t += sample_interval
    cap.release()
    return records


# ═══════════════════════════════════════════════════════════════════
# Feature extraction (handles both CNN [B,C,H,W] and ViT [B,N,D])
# ═══════════════════════════════════════════════════════════════════

def extract_features(
    model: nn.Module, backbone: str,
    frames: list[dict], device: torch.device, batch_size: int,
) -> tuple[list[torch.Tensor], int, tuple[int,int]]:
    """Returns (per_frame_patch_list, feature_dim, grid_hw)."""
    model.eval()
    all_patches: list[torch.Tensor] = []
    dim: int | None = None
    grid: tuple[int,int] | None = None

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.stack([f["tensor"] for f in frames[i:i+batch_size]]).to(device)
            out = model(batch).detach().cpu()  # CNN: (B,C,H,W)  ViT: (B,N,D)

            if backbone == "resnet18":
                B, C, H, W = out.shape
                if dim is None:
                    dim = C; grid = (H, W)
                for j in range(B):
                    flat = out[j].permute(1,2,0).reshape(-1, C)
                    all_patches.append(F.normalize(flat, dim=1))
            else:  # dinov2
                B, N, D = out.shape
                if dim is None:
                    dim = D; g = int(math.sqrt(N)); grid = (g, g)
                for j in range(B):
                    all_patches.append(out[j])  # already L2-normalized

    if dim is None or grid is None:
        raise RuntimeError("No features extracted")
    return all_patches, dim, grid


# ═══════════════════════════════════════════════════════════════════
# Coreset (unchanged)
# ═══════════════════════════════════════════════════════════════════

def build_memory_bank(
    patch_list: list[torch.Tensor], coreset_size: int,
    max_candidates: int, method: str, seed: int, chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    bank = torch.cat(patch_list, dim=0).contiguous()
    n = bank.shape[0]
    gen = torch.Generator().manual_seed(seed)
    if n > max_candidates:
        idx = torch.randperm(n, generator=gen)[:max_candidates]
        candidates = bank[idx].contiguous()
    else:
        candidates = bank
    target = min(coreset_size, candidates.shape[0])

    if method == "random":
        if candidates.shape[0] <= target:
            coreset = candidates
        else:
            idx = torch.randperm(candidates.shape[0], generator=gen)[:target]
            coreset = candidates[idx].contiguous()
    else:  # greedy
        first = int(torch.randint(0, candidates.shape[0], (1,), generator=gen).item())
        selected = [first]
        min_dist = torch.full((candidates.shape[0],), float("inf"))
        for _ in range(1, target):
            latest = candidates[selected[-1]:selected[-1]+1]
            for s in range(0, candidates.shape[0], chunk_size):
                e = min(s+chunk_size, candidates.shape[0])
                d = torch.cdist(candidates[s:e], latest).squeeze(1)
                min_dist[s:e] = torch.minimum(min_dist[s:e], d)
            selected.append(int(torch.argmax(min_dist).item()))
        coreset = candidates[torch.tensor(selected)].contiguous()

    return coreset, {
        "raw_patch_count": n,
        "candidate_patch_count": int(candidates.shape[0]),
        "coreset_patch_count": int(coreset.shape[0]),
        "coreset_method": method,
        "feature_dim": int(bank.shape[1]),
    }


# ═══════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════

def score_frames(
    frame_feats: list[torch.Tensor], memory_bank: torch.Tensor,
    device: torch.device, chunk_size: int,
) -> list[np.ndarray]:
    mem = memory_bank.to(device)
    maps: list[np.ndarray] = []
    with torch.no_grad():
        for patches in frame_feats:
            p = patches.to(device)
            mins: list[torch.Tensor] = []
            for s in range(0, p.shape[0], chunk_size):
                e = min(s+chunk_size, p.shape[0])
                mins.append(torch.cdist(p[s:e], mem).min(dim=1).values.cpu())
            maps.append(torch.cat(mins, dim=0).numpy())
    return maps


# ═══════════════════════════════════════════════════════════════════
# Heatmaps (fixed: lossless PNG, proper upsampling)
# ═══════════════════════════════════════════════════════════════════

def save_heatmap_overlay(
    roi_rgb: np.ndarray, score_map: np.ndarray, out_path: Path,
) -> None:
    h, w = roi_rgb.shape[:2]
    if score_map.ndim == 1:
        # ViT output: (N,) flat patches → reshape to grid
        g = int(math.sqrt(score_map.shape[0]))
        score_map = score_map.reshape(g, g)
    # Upsample score map to match roi crop size
    if score_map.shape[0] != h or score_map.shape[1] != w:
        up = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        up = score_map
    lo, hi = float(up.min()), float(up.max())
    norm = np.zeros_like(up, dtype=np.uint8) if math.isclose(hi, lo) \
           else ((up - lo) / (hi - lo) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(roi_rgb, 0.55, heat, 0.45, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ═══════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════

def load_labels(csv_path: Path) -> dict[float, str]:
    if not csv_path.is_file():
        return {}
    labels: dict[float, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lbl = row.get("manual_label", "").strip()
            if lbl:
                try:
                    labels[float(row["timestamp_seconds"])] = lbl
                except ValueError:
                    continue
    return labels


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 2 PatchCore video v2 (DINOv2 backbone)")
    p.add_argument("--video", type=Path,
                   default=Path("data/workstation/raw/20241101_161258.mp4"))
    p.add_argument("--output", type=Path, default=None,
                   help="Default: runs/week2_patchcore_case4_v2")
    p.add_argument("--backbone", choices=["resnet18","dinov2"], default="dinov2")
    # Tighter default ROI: side-cap band (y=380 to y=580 covers side-cap row)
    p.add_argument("--roi", type=int, nargs=4, default=(480, 380, 1190, 580),
                   metavar=("X1","Y1","X2","Y2"),
                   help="Default: tight side-cap band (480 380 1190 580)")
    p.add_argument("--bank-start", type=float, default=120.0)
    p.add_argument("--bank-end", type=float, default=260.0)
    p.add_argument("--test-start", type=float, default=260.0)
    p.add_argument("--test-end", type=float, default=0.0,
                   help="0 = full video duration")
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--image-size", type=int, default=518,
                   help="Input resolution (DINOv2 native=518, ResNet18=224)")
    p.add_argument("--batch-size", type=int, default=2,
                   help="DINOv2 needs smaller batches (default 2)")
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--coreset-size", type=int, default=1000)
    p.add_argument("--max-candidates", type=int, default=20000)
    p.add_argument("--coreset-method", choices=["random","greedy"], default="random")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--nn-chunk-size", type=int, default=2048)
    p.add_argument("--max-heatmaps", type=int, default=30)
    p.add_argument("--label-csv", type=Path,
                   default=Path("data/workstation/manifests/week2_case4_manual_label_template.csv"))
    p.add_argument("--top-anomaly-n", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    output_dir = args.output or Path("runs") / "week2_patchcore_case4_v2"
    heatmap_dir = output_dir / "heatmaps"
    for d in [output_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    log_lines = [
        "# Week 2 PatchCore Video v2 Run Log",
        f"- Backbone: **{args.backbone}**",
        f"- ROI: {tuple(args.roi)}",
        f"- Bank Normal: t={args.bank_start}-{args.bank_end}s",
        f"- Test Stream: t={args.test_start}-{args.test_end if args.test_end>0 else 'end'}s",
        f"- Sample interval: {args.sample_interval}s",
        f"- Coreset: {args.coreset_size} ({args.coreset_method})",
        f"- Image size: {args.image_size}, Feature dim: {args.feature_dim}",
        "",
    ]

    result: dict = {
        "algorithm": f"PatchCore-video-v2-{args.backbone}",
        "video": str(args.video),
        "roi": list(args.roi),
        "backbone": args.backbone,
        "status": "failed",
        "command": "python " + " ".join(sys.argv),
    }

    try:
        # ── 1. Load labels ──
        labels = load_labels(args.label_csv)
        log_lines.append(f"- Manual labels loaded: {len(labels)}")

        # ── 2. Video metadata ──
        cap = cv2.VideoCapture(str(args.video))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(video_fps, 1)
        cap.release()
        test_end = args.test_end if args.test_end > 0 else duration

        print(f"Video duration: {duration:.1f}s, FPS: {video_fps:.2f}")
        print(f"Backbone: {args.backbone}, ROI: {args.roi}")
        print(f"Bank: t={args.bank_start}-{args.bank_end}s, Test: t={args.test_start}-{test_end:.1f}s")

        # ── 3. Build model ──
        model = build_backbone(args.backbone, args.image_size, args.feature_dim, args.seed)
        model = model.to(device).eval()

        # ── 4. Bank Normal frames ──
        bank_frames = iter_video_frames(
            args.video, tuple(args.roi), video_fps, args.sample_interval,
            args.bank_start, args.bank_end, args.image_size,
        )
        print(f"Bank Normal frames: {len(bank_frames)}")
        if not bank_frames:
            raise RuntimeError("No Bank Normal frames — check ROI / time window")

        bank_feats, feat_dim, grid_hw = extract_features(
            model, args.backbone, bank_frames, device, args.batch_size,
        )
        memory_bank, mem_info = build_memory_bank(
            bank_feats, args.coreset_size, args.max_candidates,
            args.coreset_method, args.seed, args.nn_chunk_size,
        )
        memory_path = output_dir / "memory_bank.pt"
        torch.save({
            "memory_bank": memory_bank, "feature_dim": feat_dim,
            "grid_hw": grid_hw, "memory_info": mem_info,
            "roi": args.roi, "backbone": args.backbone, "image_size": args.image_size,
        }, memory_path)
        print(f"Memory bank: {mem_info['coreset_patch_count']} patches (raw {mem_info['raw_patch_count']})")

        # ── 5. Test Stream frames ──
        test_frames = iter_video_frames(
            args.video, tuple(args.roi), video_fps, args.sample_interval,
            args.test_start, test_end, args.image_size,
        )
        print(f"Test Stream frames: {len(test_frames)}")
        test_feats, _, _ = extract_features(
            model, args.backbone, test_frames, device, args.batch_size,
        )
        score_maps = score_frames(test_feats, memory_bank, device, args.nn_chunk_size)
        print(f"Scored {len(score_maps)} frames")

        # ── 6. Aggregate ──
        frame_results: list[dict] = []
        for i, (frec, smap) in enumerate(zip(test_frames, score_maps)):
            if smap.ndim == 1:
                smap = smap.reshape(grid_hw)
            smap_blur = cv2.GaussianBlur(smap, (5, 5), 0)
            img_score = float(smap_blur.max())
            t = frec["timestamp"]
            nearest_label = None; nearest_label_t = None
            min_dist = 999.0
            for lt, lbl in labels.items():
                d = abs(t - lt)
                if d < min_dist:
                    min_dist = d; nearest_label = lbl; nearest_label_t = lt
            frame_results.append({
                "timestamp": t, "frame_idx": frec["frame_idx"],
                "anomaly_score": img_score,
                "score_mean": float(smap_blur.mean()),
                "score_std": float(smap_blur.std()),
                "manual_label": nearest_label if min_dist < args.sample_interval * 2 else None,
            })

        timeline = sorted(frame_results, key=lambda r: r["anomaly_score"], reverse=True)

        # ── 7. Heatmaps ──
        saved = 0
        for entry in timeline[:args.top_anomaly_n]:
            if saved >= args.max_heatmaps:
                break
            idx = next(i for i, r in enumerate(frame_results)
                       if abs(r["timestamp"]-entry["timestamp"]) < 0.01)
            smap = score_maps[idx]
            if smap.ndim == 1:
                smap = smap.reshape(grid_hw)
            smap_blur = cv2.GaussianBlur(smap, (5,5), 0)
            t_str = f"{entry['timestamp']:.1f}s".replace(".","p")
            lbl = entry.get("manual_label") or "unknown"
            save_heatmap_overlay(
                test_frames[idx]["roi_array"], smap_blur,
                heatmap_dir / f"t{t_str}_score{entry['anomaly_score']:.4f}_{lbl}.png",
            )
            saved += 1

        # ── 8. Hit-rate ──
        anomaly_ts_set = {t for t, lbl in labels.items() if lbl == "anomaly"}
        hit_details = []
        for k in [5, 10, 20, 30]:
            hits = 0
            for r in timeline[:k]:
                for at in anomaly_ts_set:
                    if abs(r["timestamp"] - at) < args.sample_interval * 1.5:
                        hits += 1; break
            hit_details.append({
                "top_k": k, "hits": hits,
                "total_anomaly_labels": len(anomaly_ts_set),
                "hit_rate": round(hits / max(len(anomaly_ts_set), 1), 3),
            })

        # ── 9. Write outputs ──
        scores_csv = output_dir / "frame_scores.csv"
        with open(scores_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp","frame_idx","anomaly_score","score_mean",
                "score_std","manual_label"])
            w.writeheader()
            for row in sorted(frame_results, key=lambda r: r["timestamp"]):
                w.writerow({k: row.get(k, "") for k in w.fieldnames})

        timeline_csv = output_dir / "anomaly_timeline.csv"
        with open(timeline_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "rank","timestamp","frame_idx","anomaly_score",
                "score_mean","score_std","manual_label"])
            w.writeheader()
            for rank, entry in enumerate(timeline, 1):
                w.writerow({**entry, "rank": rank})

        runtime = round(time.perf_counter() - start_time, 3)
        result.update({
            "status": "success", "runtime_seconds": runtime,
            "max_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device)/(1024**2),3)
                if use_cuda else None,
            "bank_normal_frames": len(bank_frames),
            "test_frames": len(test_frames),
            "bank_normal_window": [args.bank_start, args.bank_end],
            "test_window": [args.test_start, round(test_end, 1)],
            "feature_grid_hw": list(grid_hw),
            "memory_bank_info": mem_info,
            "top_k_anomaly_hits": hit_details,
            "heatmaps_saved": saved,
            "output_dir": str(output_dir),
            "error": None,
        })
        (output_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        log_lines.extend([
            f"- Status: **success**",
            f"- Bank frames: {len(bank_frames)}, Test frames: {len(test_frames)}",
            f"- Memory bank: {mem_info['coreset_patch_count']} patches",
            f"- Feature grid: {grid_hw}",
            f"- Runtime: {runtime}s",
            f"- GPU peak: {result['max_gpu_mem_mb']} MB",
            f"- Top-10 hit rate: {hit_details[1]['hit_rate']}",
            f"- Heatmaps: {saved}",
        ])
        (output_dir / "run_log.md").write_text("\n".join(log_lines)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        runtime = round(time.perf_counter() - start_time, 3)
        result["runtime_seconds"] = runtime
        result["error"] = f"{type(exc).__name__}: {exc}"
        (output_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log_lines.append(f"- Status: **failed**\n- Error: {result['error']}")
        (output_dir / "run_log.md").write_text("\n".join(log_lines)+"\n", encoding="utf-8")
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
