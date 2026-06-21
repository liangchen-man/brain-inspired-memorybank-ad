#!/usr/bin/env python3
"""Teaching AnomalyDINO-style runner for MVTec AD.

This runner uses DINO-family ViT patch tokens from timm as a normal patch
memory bank, then scores test patch tokens by nearest-neighbor distance. It is
for a course-level, reproducible AnomalyDINO-style attempt and is not an
official AnomalyDINO repository reproduction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from run_patchcore import (
    ImagePathDataset,
    build_coreset,
    build_test_samples,
    build_train_samples,
    make_channel_index,
    read_mask,
    safe_auc,
    save_visuals,
    validate_category,
)


class TimmPatchTokenModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: int,
        pretrained: bool,
        feature_dim: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,
        )
        self.feature_dim = feature_dim
        self.seed = seed
        self.channel_idx: torch.Tensor | None = None
        self.token_grid: tuple[int, int] | None = None

    def _tokens_from_features(self, features: torch.Tensor | dict) -> torch.Tensor:
        if isinstance(features, dict):
            for key in ["x_norm_patchtokens", "x_patchtokens", "patch_tokens"]:
                value = features.get(key)
                if value is not None:
                    return value
            value = features.get("x_norm")
            if value is None:
                value = features.get("x")
            if value is None:
                raise RuntimeError(f"Unsupported timm feature dict keys: {sorted(features.keys())}")
            features = value

        if not torch.is_tensor(features) or features.ndim != 3:
            raise RuntimeError(f"Expected ViT token tensor [B, N, C], got {type(features)}")

        token_count = features.shape[1]
        grid = int(math.sqrt(token_count))
        if grid * grid == token_count:
            return features

        no_prefix = token_count - 1
        grid = int(math.sqrt(no_prefix))
        if grid * grid == no_prefix:
            return features[:, 1:, :]

        raise RuntimeError(f"Cannot infer square patch grid from token count {token_count}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)
        tokens = self._tokens_from_features(features)
        b, n, c = tokens.shape
        grid = int(math.sqrt(n))
        if grid * grid != n:
            raise RuntimeError(f"Patch token count {n} is not a square grid.")

        if self.channel_idx is None:
            self.channel_idx = make_channel_index(c, self.feature_dim, self.seed)
            self.token_grid = (grid, grid)

        reduced = tokens[:, :, self.channel_idx.to(tokens.device)]
        reduced = F.normalize(reduced, dim=2)
        return reduced


def extract_patch_bank(
    model: TimmPatchTokenModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    model.eval()
    patches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            tokens = model(batch["image"].to(device)).detach().cpu()
            b, n, c = tokens.shape
            patches.append(tokens.reshape(b * n, c))

    if not patches or model.channel_idx is None or model.token_grid is None:
        raise RuntimeError("No DINO patch tokens extracted.")
    return torch.cat(patches, dim=0).contiguous(), model.channel_idx, model.token_grid


def nearest_patch_distances(
    patches: torch.Tensor,
    memory_bank: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, patches.shape[0], chunk_size):
        end = min(start + chunk_size, patches.shape[0])
        dist = torch.cdist(patches[start:end], memory_bank)
        chunks.append(dist.min(dim=1).values.cpu())
    return torch.cat(chunks, dim=0)


def score_batch(
    model: TimmPatchTokenModel,
    images: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
    nn_chunk_size: int,
) -> torch.Tensor:
    with torch.no_grad():
        tokens = model(images.to(device)).detach().cpu()
    b, n, c = tokens.shape
    h, w = model.token_grid or (int(math.sqrt(n)), int(math.sqrt(n)))
    distances = nearest_patch_distances(tokens.reshape(b * n, c), memory_bank, nn_chunk_size)
    return distances.reshape(b, h, w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teaching AnomalyDINO-style anomaly detection on MVTec AD.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--coreset-size", type=int, default=2000)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--coreset-method", choices=["random", "greedy"], default="random")
    parser.add_argument("--max-heatmaps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nn-chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    output_dir = args.output or Path("runs") / f"anomalydino_{args.category}"
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    for path in [output_dir, heatmap_dir, overlay_dir]:
        path.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    run_log_path = output_dir / "run_log.md"
    memory_path = output_dir / "memory_bank.pt"
    command = "python " + " ".join(__import__("sys").argv)

    result = {
        "algorithm": "AnomalyDINO-teaching",
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
        "backbone": args.model_name,
        "heatmap_paths": [],
        "overlay_paths": [],
        "output_dir": str(output_dir),
        "error": None,
        "notes": "Teaching AnomalyDINO-style DINO/ViT patch token memory. pretrained=False means representation-limited offline attempt.",
    }
    log_lines = [
        f"# AnomalyDINO Run Log: {args.category}",
        "",
        f"- Command: `{command}`",
        f"- Output: `{output_dir}`",
        f"- Model: {args.model_name}",
        f"- Pretrained: {args.pretrained}",
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

        model = TimmPatchTokenModel(
            model_name=args.model_name,
            image_size=args.image_size,
            pretrained=args.pretrained,
            feature_dim=args.feature_dim,
            seed=args.seed,
        ).to(device).eval()

        patch_bank, channel_idx, token_grid = extract_patch_bank(model, train_loader, device)
        memory_bank, memory_info = build_coreset(
            patch_bank=patch_bank,
            method=args.coreset_method,
            coreset_size=args.coreset_size,
            max_candidates=args.max_candidates,
            seed=args.seed,
            chunk_size=args.nn_chunk_size,
        )
        memory_info.update(
            {
                "model_name": args.model_name,
                "pretrained": bool(args.pretrained),
                "token_grid": list(token_grid),
            }
        )
        torch.save(
            {
                "memory_bank": memory_bank,
                "channel_idx": channel_idx,
                "token_grid": token_grid,
                "memory_info": memory_info,
                "category": args.category,
                "image_size": args.image_size,
            },
            memory_path,
        )
        memory_info["memory_bank_file"] = str(memory_path)
        memory_info["memory_bank_file_bytes"] = memory_path.stat().st_size

        image_labels: list[int] = []
        image_scores: list[float] = []
        pixel_labels: list[int] = []
        pixel_scores: list[float] = []
        saved_heatmaps = 0

        for idx, batch in enumerate(test_loader):
            score_map = score_batch(
                model=model,
                images=batch["image"],
                memory_bank=memory_bank,
                device=device,
                nn_chunk_size=args.nn_chunk_size,
            )[0].numpy()
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
                f"- Token grid: {memory_info['token_grid']}",
                f"- Raw patch count: {memory_info['raw_patch_count']}",
                f"- Coreset patch count: {memory_info['coreset_patch_count']}",
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
