from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
REPORT_DIR = ROOT / "report"
FIGURES_PATH = REPORT_DIR / "figures_index.md"


def load_result(result_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def section_for_run(run_dir: Path) -> list[str]:
    result_path = run_dir / "results.json"
    data = load_result(result_path)
    if data is None:
        return [
            f"## {run_dir.name}",
            "",
            "- Status: 未完成",
            "- Reason: results.json not found or unreadable.",
            "",
        ]

    heatmaps = data.get("heatmap_paths") or []
    overlays = data.get("overlay_paths") or []
    lines = [
        f"## {run_dir.name}",
        "",
        f"- Algorithm: {data.get('algorithm', 'unknown')}",
        f"- Category: {data.get('category', 'unknown')}",
        f"- Status: {data.get('status', 'unknown')}",
        f"- Image AUROC: {data.get('image_auroc', 'null')}",
        f"- Pixel AUROC: {data.get('pixel_auroc', 'null')}",
        f"- Heatmaps: {len(heatmaps)}",
        f"- Overlays: {len(overlays)}",
        "",
    ]

    preview_count = min(3, len(heatmaps), len(overlays))
    if preview_count == 0:
        lines.extend(["- Preview: 未生成 heatmap/overlay。", ""])
        return lines

    lines.extend(["| # | Heatmap | Overlay |", "|---:|---|---|"])
    for index in range(preview_count):
        lines.append(f"| {index + 1} | `{rel(heatmaps[index])}` | `{rel(overlays[index])}` |")
    lines.append("")
    return lines


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figures Index",
        "",
        "This file indexes generated heatmaps and overlays for the experiment report.",
        "Only files produced under `runs/` are listed; missing experiments are marked as 未完成.",
        "",
    ]

    expected = [
        "padim_carpet",
        "padim_bottle",
        "patchcore_carpet",
        "patchcore_bottle",
        "spade_carpet",
        "anomalydino_carpet",
    ]
    for run_name in expected:
        lines.extend(section_for_run(RUNS_DIR / run_name))

    FIGURES_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {FIGURES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
