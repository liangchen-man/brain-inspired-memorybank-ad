#!/usr/bin/env python3
"""Teaching PatchCore runner for MVTec AD.

This compact implementation is designed for the course project. It extracts
normal patch features, keeps a small representative memory bank with a simple
coreset strategy, and scores test patches by nearest-neighbor distance.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int) -> None:
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        original_size = image.size
        return {
            "image": self.transform(image),
            "path": str(sample.path),
            "label": sample.label,
            "defect_type": sample.defect_type,
            "mask_path": str(sample.mask_path) if sample.mask_path else "",
            "original_w": original_size[0],
            "original_h": original_size[1],
        }


class ResNet18Layer2(nn.Module):
    def __init__(self, weights: str) -> None:
        super().__init__()
        if weights == "imagenet":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif weights == "none":
            backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported weights mode: {weights}")

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def iter_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_train_samples(category_root: Path) -> list[Sample]:
    return [Sample(path=p, label=0, defect_type="good") for p in iter_images(category_root / "train" / "good")]


def build_test_samples(category_root: Path) -> list[Sample]:
    test_root = category_root / "test"
    gt_root = category_root / "ground_truth"
    samples: list[Sample] = []
    for subdir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        defect_type = subdir.name
        label = 0 if defect_type == "good" else 1
        for image_path in iter_images(subdir):
            mask_path = None
            if label == 1:
                candidate = gt_root / defect_type / f"{image_path.stem}_mask.png"
                if candidate.is_file():
                    mask_path = candidate
            samples.append(Sample(image_path, label, defect_type, mask_path))
    return samples


def validate_category(category_root: Path) -> None:
    required = [
        category_root,
        category_root / "train" / "good",
        category_root / "test" / "good",
        category_root / "ground_truth",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required MVTec paths: " + "; ".join(missing))


def make_channel_index(channels: int, feature_dim: int, seed: int) -> torch.Tensor:
    if feature_dim > channels:
        feature_dim = channels
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(channels, generator=gen)[:feature_dim]


def extract_patch_bank(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    model.eval()
    patches: list[torch.Tensor] = []
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
            b, c, h, w = reduced.shape
            flat = reduced.permute(0, 2, 3, 1).reshape(b * h * w, c)
            patches.append(flat)

    if not patches or channel_idx is None or feature_hw is None:
        raise RuntimeError("No patch features extracted.")
    bank = torch.cat(patches, dim=0).contiguous()
    bank = F.normalize(bank, dim=1)
    return bank, channel_idx, feature_hw


def random_candidate_subset(bank: torch.Tensor, max_candidates: int, seed: int) -> torch.Tensor:
    if bank.shape[0] <= max_candidates:
        return bank
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(bank.shape[0], generator=gen)[:max_candidates]
    return bank[idx].contiguous()


def random_coreset(bank: torch.Tensor, coreset_size: int, seed: int) -> torch.Tensor:
    if bank.shape[0] <= coreset_size:
        return bank
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(bank.shape[0], generator=gen)[:coreset_size]
    return bank[idx].contiguous()


def greedy_coreset(
    bank: torch.Tensor,
    coreset_size: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    if bank.shape[0] <= coreset_size:
        return bank

    gen = torch.Generator().manual_seed(seed)
    first = int(torch.randint(0, bank.shape[0], (1,), generator=gen).item())
    selected = [first]
    min_dist = torch.full((bank.shape[0],), float("inf"))

    for _ in range(1, coreset_size):
        latest = bank[selected[-1] : selected[-1] + 1]
        for start in range(0, bank.shape[0], chunk_size):
            end = min(start + chunk_size, bank.shape[0])
            dist = torch.cdist(bank[start:end], latest).squeeze(1)
            min_dist[start:end] = torch.minimum(min_dist[start:end], dist)
        selected.append(int(torch.argmax(min_dist).item()))

    return bank[torch.tensor(selected)].contiguous()


def build_coreset(
    patch_bank: torch.Tensor,
    method: str,
    coreset_size: int,
    max_candidates: int,
    seed: int,
    chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    candidates = random_candidate_subset(patch_bank, max_candidates, seed)
    target = min(coreset_size, candidates.shape[0])
    if method == "greedy":
        coreset = greedy_coreset(candidates, target, seed, chunk_size)
    elif method == "random":
        coreset = random_coreset(candidates, target, seed)
    else:
        raise ValueError(f"Unsupported coreset method: {method}")
    info = {
        "raw_patch_count": int(patch_bank.shape[0]),
        "candidate_patch_count": int(candidates.shape[0]),
        "coreset_patch_count": int(coreset.shape[0]),
        "coreset_method": method,
        "feature_dim": int(patch_bank.shape[1]),
    }
    return coreset, info


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
    model: nn.Module,
    images: torch.Tensor,
    channel_idx: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
    nn_chunk_size: int,
) -> torch.Tensor:
    feats = model(images.to(device))[:, channel_idx.to(device), :, :].detach().cpu()
    b, c, h, w = feats.shape
    patches = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
    patches = F.normalize(patches, dim=1)
    distances = nearest_patch_distances(patches, memory_bank, nn_chunk_size)
    return distances.reshape(b, h, w)


def normalize_map(score_map: np.ndarray) -> np.ndarray:
    lo = float(score_map.min())
    hi = float(score_map.max())
    if math.isclose(hi, lo):
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - lo) / (hi - lo)).astype(np.float32)


def read_mask(mask_path: str, size_hw: tuple[int, int]) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = size_hw
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def save_visuals(
    image_path: str,
    score_map: np.ndarray,
    heatmap_dir: Path,
    overlay_dir: Path,
    sample_id: str,
) -> tuple[str, str]:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = image_bgr.shape[:2]
    resized = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_CUBIC)
    norm = normalize_map(resized)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.55, heat, 0.45, 0)
    heat_path = heatmap_dir / f"{sample_id}_heatmap.png"
    overlay_path = overlay_dir / f"{sample_id}_overlay.png"
    cv2.imwrite(str(heat_path), heat)
    cv2.imwrite(str(overlay_path), overlay)
    return str(heat_path), str(overlay_path)


def safe_auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    labels_list = list(labels)
    scores_list = list(scores)
    if len(set(labels_list)) < 2:
        return None
    return float(roc_auc_score(labels_list, scores_list))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teaching PatchCore on MVTec AD.")
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--coreset-size", type=int, default=2000)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--coreset-method", choices=["random", "greedy"], default="random")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max-heatmaps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nn-chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    output_dir = args.output or Path("runs") / f"patchcore_{args.category}"
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    for path in [output_dir, heatmap_dir, overlay_dir]:
        path.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    run_log_path = output_dir / "run_log.md"
    memory_path = output_dir / "memory_bank.pt"
    command = "python " + " ".join(__import__("sys").argv)

    result = {
        "algorithm": "PatchCore-teaching",
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
        "notes": "Teaching PatchCore using ResNet18 patch memory with coreset.",
    }
    log_lines = [
        f"# PatchCore Run Log: {args.category}",
        "",
        f"- Command: `{command}`",
        f"- Output: `{output_dir}`",
        f"- Weights: {args.weights}",
        f"- Coreset method: {args.coreset_method}",
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
        patch_bank, channel_idx, feature_hw = extract_patch_bank(
            model=model,
            loader=train_loader,
            device=device,
            feature_dim=args.feature_dim,
            seed=args.seed,
        )
        memory_bank, memory_info = build_coreset(
            patch_bank=patch_bank,
            method=args.coreset_method,
            coreset_size=args.coreset_size,
            max_candidates=args.max_candidates,
            seed=args.seed,
            chunk_size=args.nn_chunk_size,
        )
        torch.save(
            {
                "memory_bank": memory_bank,
                "channel_idx": channel_idx,
                "feature_hw": feature_hw,
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

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                score_map = score_batch(
                    model=model,
                    images=batch["image"],
                    channel_idx=channel_idx,
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
                upsampled = cv2.resize(
                    score_map,
                    (original_w, original_h),
                    interpolation=cv2.INTER_CUBIC,
                )
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
