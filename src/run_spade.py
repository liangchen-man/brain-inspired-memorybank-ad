#!/usr/bin/env python3
"""Teaching SPADE-style runner for MVTec AD.

This compact implementation is for the course project. It stores normal image
features and their spatial feature maps, retrieves nearest normal images for a
test sample, and turns the local feature discrepancy into an anomaly heatmap.
It is not an official SPADE repository reproduction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from run_patchcore import (
    ImagePathDataset,
    ResNet18Layer2,
    build_test_samples,
    build_train_samples,
    make_channel_index,
    read_mask,
    safe_auc,
    save_visuals,
    validate_category,
)


def extract_spade_memory(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    model.eval()
    global_vectors: list[torch.Tensor] = []
    spatial_maps: list[torch.Tensor] = []
    channel_idx: torch.Tensor | None = None
    feature_hw: tuple[int, int] | None = None

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feats = model(images)
            if channel_idx is None:
                channel_idx = make_channel_index(feats.shape[1], feature_dim, seed)
                feature_hw = (feats.shape[2], feats.shape[3])

            reduced = feats[:, channel_idx.to(feats.device), :, :].detach().cpu()
            pooled = F.adaptive_avg_pool2d(reduced, output_size=1).flatten(1)
            global_vectors.append(F.normalize(pooled, dim=1))
            spatial_maps.append(F.normalize(reduced, dim=1))

    if not global_vectors or channel_idx is None or feature_hw is None:
        raise RuntimeError("No normal features extracted.")

    return (
        torch.cat(global_vectors, dim=0).contiguous(),
        torch.cat(spatial_maps, dim=0).contiguous(),
        channel_idx,
        feature_hw,
    )


def score_one_image(
    model: nn.Module,
    image: torch.Tensor,
    channel_idx: torch.Tensor,
    global_memory: torch.Tensor,
    spatial_memory: torch.Tensor,
    device: torch.device,
    k_neighbors: int,
) -> tuple[np.ndarray, list[int]]:
    with torch.no_grad():
        feats = model(image.to(device))[:, channel_idx.to(device), :, :].detach().cpu()
    feats = F.normalize(feats, dim=1)
    query_global = F.normalize(F.adaptive_avg_pool2d(feats, output_size=1).flatten(1), dim=1)
    global_dist = torch.cdist(query_global, global_memory).squeeze(0)
    k = min(k_neighbors, global_memory.shape[0])
    neighbor_idx = torch.topk(global_dist, k=k, largest=False).indices

    query_map = feats[0]
    neighbor_maps = spatial_memory[neighbor_idx]
    distances = torch.sqrt(torch.clamp(((neighbor_maps - query_map.unsqueeze(0)) ** 2).sum(dim=1), min=0.0))
    score_map = distances.min(dim=0).values.numpy()
    return score_map, [int(i) for i in neighbor_idx.tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teaching SPADE-style anomaly detection on MVTec AD.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max-heatmaps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    output_dir = args.output or Path("runs") / f"spade_{args.category}"
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    for path in [output_dir, heatmap_dir, overlay_dir]:
        path.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    run_log_path = output_dir / "run_log.md"
    memory_path = output_dir / "memory_bank.pt"
    command = "python " + " ".join(__import__("sys").argv)

    result = {
        "algorithm": "SPADE-teaching",
        "category": args.category,
        "status": "failed",
        "command": command,
        "runtime_seconds": None,
        "max_gpu_mem_mb": None,
        "image_auroc": None,
        "pixel_auroc": None,
        "pro_score": None,
        "memory_bank_size": None,
        "memory_bank_path": str(memory_path),
        "gaussian_param_path": None,
        "reference_image_count": None,
        "backbone": "resnet18_layer2",
        "heatmap_paths": [],
        "overlay_paths": [],
        "output_dir": str(output_dir),
        "error": None,
        "notes": "Teaching SPADE-style normal image nearest-neighbor memory.",
    }
    log_lines = [
        f"# SPADE Run Log: {args.category}",
        "",
        f"- Command: `{command}`",
        f"- Output: `{output_dir}`",
        f"- Weights: {args.weights}",
        f"- K neighbors: {args.k_neighbors}",
        "",
    ]

    try:
        category_root = args.data_root / args.category
        validate_category(category_root)
        train_samples = build_train_samples(category_root)
        test_samples = build_test_samples(category_root)
        if not train_samples:
            raise RuntimeError("No train/good images found.")
        if not test_samples:
            raise RuntimeError("No test images found.")

        use_cuda = args.device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        train_loader = DataLoader(
            ImagePathDataset(train_samples, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            ImagePathDataset(test_samples, args.image_size),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = ResNet18Layer2(args.weights).to(device).eval()
        global_memory, spatial_memory, channel_idx, feature_hw = extract_spade_memory(
            model=model,
            loader=train_loader,
            device=device,
            feature_dim=args.feature_dim,
            seed=args.seed,
        )
        torch.save(
            {
                "global_memory": global_memory,
                "spatial_memory": spatial_memory,
                "channel_idx": channel_idx,
                "feature_hw": feature_hw,
                "category": args.category,
                "image_size": args.image_size,
                "k_neighbors": args.k_neighbors,
            },
            memory_path,
        )

        memory_info = {
            "global_memory_shape": list(global_memory.shape),
            "spatial_memory_shape": list(spatial_memory.shape),
            "feature_hw": list(feature_hw),
            "k_neighbors": int(args.k_neighbors),
            "feature_dim": int(global_memory.shape[1]),
            "memory_bank_file": str(memory_path),
            "memory_bank_file_bytes": memory_path.stat().st_size,
        }

        image_labels: list[int] = []
        image_scores: list[float] = []
        pixel_labels: list[int] = []
        pixel_scores: list[float] = []
        saved_heatmaps = 0

        for idx, batch in enumerate(test_loader):
            score_map, _ = score_one_image(
                model=model,
                image=batch["image"],
                channel_idx=channel_idx,
                global_memory=global_memory,
                spatial_memory=spatial_memory,
                device=device,
                k_neighbors=args.k_neighbors,
            )
            score_map = cv2.GaussianBlur(score_map, (5, 5), 0)
            image_score = float(score_map.max())
            label = int(batch["label"].item())
            image_labels.append(label)
            image_scores.append(image_score)

            original_h = int(batch["original_h"].item())
            original_w = int(batch["original_w"].item())
            upsampled = cv2.resize(score_map, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            mask = read_mask(batch["mask_path"][0], (original_h, original_w))
            if mask is not None:
                pixel_labels.extend(mask.reshape(-1).tolist())
                pixel_scores.extend(upsampled.reshape(-1).tolist())

            if saved_heatmaps < args.max_heatmaps:
                image_path = batch["path"][0]
                defect_type = batch["defect_type"][0]
                sample_id = f"{idx:04d}_{defect_type}_{Path(image_path).stem}"
                heat_path, overlay_path = save_visuals(
                    image_path=image_path,
                    score_map=upsampled,
                    heatmap_dir=heatmap_dir,
                    overlay_dir=overlay_dir,
                    sample_id=sample_id,
                )
                result["heatmap_paths"].append(heat_path)
                result["overlay_paths"].append(overlay_path)
                saved_heatmaps += 1

        result["image_auroc"] = safe_auc(image_labels, image_scores)
        result["pixel_auroc"] = safe_auc(pixel_labels, pixel_scores) if pixel_labels else None
        result["runtime_seconds"] = round(time.perf_counter() - start, 3)
        if use_cuda:
            result["max_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024**2), 3)
        result["memory_bank_size"] = memory_info
        result["reference_image_count"] = len(train_samples)
        result["status"] = "success"

        log_lines.extend(
            [
                "- Status: success",
                f"- Device: {device}",
                f"- Train images: {len(train_samples)}",
                f"- Test images: {len(test_samples)}",
                f"- Global memory shape: {memory_info['global_memory_shape']}",
                f"- Spatial memory shape: {memory_info['spatial_memory_shape']}",
                f"- Image AUROC: {result['image_auroc']}",
                f"- Pixel AUROC: {result['pixel_auroc']}",
                f"- Runtime seconds: {result['runtime_seconds']}",
                f"- Max GPU memory MB: {result['max_gpu_mem_mb']}",
                f"- Heatmaps saved: {len(result['heatmap_paths'])}",
            ]
        )
    except Exception as exc:
        result["runtime_seconds"] = round(time.perf_counter() - start, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"
        log_lines.extend(["- Status: failed", f"- Error: {result['error']}"])

    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    run_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
