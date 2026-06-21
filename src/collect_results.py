from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

EXPECTED_RUNS = [
    ("PaDiM-teaching", "carpet", "padim_carpet"),
    ("PaDiM-teaching", "bottle", "padim_bottle"),
    ("PatchCore-teaching", "carpet", "patchcore_carpet"),
    ("PatchCore-teaching", "bottle", "patchcore_bottle"),
    ("SPADE", "carpet", "spade_carpet"),
    ("AnomalyDINO", "carpet", "anomalydino_carpet"),
]

FIELDS = [
    "algorithm",
    "category",
    "run_dir",
    "status",
    "image_auroc",
    "pixel_auroc",
    "pro_score",
    "runtime_seconds",
    "max_gpu_mem_mb",
    "memory_bank_file_bytes",
    "memory_bank_detail",
    "reference_image_count",
    "heatmap_count",
    "overlay_count",
    "output_dir",
    "command",
    "error",
    "notes",
]


def value_or_null(value: Any) -> Any:
    return "null" if value is None else value


def memory_bytes(memory_bank_size: Any) -> Any:
    if isinstance(memory_bank_size, dict):
        return value_or_null(memory_bank_size.get("memory_bank_file_bytes"))
    return "null"


def compact_memory_detail(memory_bank_size: Any) -> str:
    if not isinstance(memory_bank_size, dict):
        return "null"

    keys = [
        "model_name",
        "pretrained",
        "token_grid",
        "train_feature_shape",
        "raw_patch_count",
        "candidate_patch_count",
        "coreset_patch_count",
        "coreset_method",
        "feature_dim",
        "mean_shape",
        "inv_cov_shape",
    ]
    detail = {key: memory_bank_size[key] for key in keys if key in memory_bank_size}
    return json.dumps(detail, ensure_ascii=False, separators=(",", ":")) if detail else "null"


def row_from_result(default_algorithm: str, category: str, run_dir: str) -> dict[str, Any]:
    result_path = RUNS_DIR / run_dir / "results.json"
    if not result_path.exists():
        return {
            "algorithm": default_algorithm,
            "category": category,
            "run_dir": run_dir,
            "status": "未完成",
            "image_auroc": "null",
            "pixel_auroc": "null",
            "pro_score": "null",
            "runtime_seconds": "null",
            "max_gpu_mem_mb": "null",
            "memory_bank_file_bytes": "null",
            "memory_bank_detail": "null",
            "reference_image_count": "null",
            "heatmap_count": 0,
            "overlay_count": 0,
            "output_dir": str(RUNS_DIR / run_dir),
            "command": "未运行",
            "error": "未完成",
            "notes": "No results.json found; do not treat as a completed experiment.",
        }

    data = json.loads(result_path.read_text(encoding="utf-8"))
    memory_bank_size = data.get("memory_bank_size")
    return {
        "algorithm": data.get("algorithm", default_algorithm),
        "category": data.get("category", category),
        "run_dir": run_dir,
        "status": data.get("status", "unknown"),
        "image_auroc": value_or_null(data.get("image_auroc")),
        "pixel_auroc": value_or_null(data.get("pixel_auroc")),
        "pro_score": value_or_null(data.get("pro_score")),
        "runtime_seconds": value_or_null(data.get("runtime_seconds")),
        "max_gpu_mem_mb": value_or_null(data.get("max_gpu_mem_mb")),
        "memory_bank_file_bytes": memory_bytes(memory_bank_size),
        "memory_bank_detail": compact_memory_detail(memory_bank_size),
        "reference_image_count": value_or_null(data.get("reference_image_count")),
        "heatmap_count": len(data.get("heatmap_paths") or []),
        "overlay_count": len(data.get("overlay_paths") or []),
        "output_dir": data.get("output_dir", str(RUNS_DIR / run_dir)),
        "command": data.get("command", "null"),
        "error": value_or_null(data.get("error")),
        "notes": data.get("notes", ""),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_from_result(*item) for item in EXPECTED_RUNS]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
