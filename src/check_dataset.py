#!/usr/bin/env python3
"""只读检查 MVTec AD 数据集解压状态。不修改、不删除、不移动任何文件。"""

import datetime
from pathlib import Path

ROOT = Path(r"E:\leinaozuoye")
DATA = ROOT / "data" / "mvtec"
REPORT = ROOT / "report" / "dataset_check.md"

TARGET = ["carpet", "grid", "leather", "tile", "wood", "bottle", "capsule", "hazelnut"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}


def imcount(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def check_class(root: Path, cls: str) -> dict:
    d = {
        "cls": cls,
        "train_good": 0,
        "test_good": 0,
        "test_defect": 0,
        "gt_mask": 0,
        "test_types": [],
        "gt_types": [],
        "missing": [],
    }
    cd = root / cls
    if not cd.is_dir():
        d["missing"].append("class dir")
        return d

    # train/good
    tg = cd / "train" / "good"
    if tg.is_dir():
        d["train_good"] = imcount(tg)
    else:
        d["missing"].append("train/good")

    # test/good
    teg = cd / "test" / "good"
    if teg.is_dir():
        d["test_good"] = imcount(teg)
    else:
        d["missing"].append("test/good")

    # test/defect_types
    td = cd / "test"
    if td.is_dir():
        for sub in sorted(td.iterdir()):
            if sub.is_dir() and sub.name != "good":
                cnt = imcount(sub)
                d["test_types"].append((sub.name, cnt))
                d["test_defect"] += cnt
    else:
        d["missing"].append("test/")

    # ground_truth/defect_types
    gt = cd / "ground_truth"
    if gt.is_dir():
        for sub in sorted(gt.iterdir()):
            if sub.is_dir():
                cnt = imcount(sub)
                d["gt_types"].append((sub.name, cnt))
                d["gt_mask"] += cnt
    else:
        d["missing"].append("ground_truth/")

    return d


def generate_report(results: list[dict]) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# MVTec AD Dataset Check Report")
    lines.append(f"")
    lines.append(f"- **Check time**: {now}")
    lines.append(f"- **Data root**: `{DATA}`")
    lines.append(f"- **Data root exists**: {DATA.is_dir()}")
    lines.append(f"")

    # Overall status
    complete = [r for r in results if not r["missing"]]
    incomplete = [r for r in results if r["missing"]]
    lines.append(f"## Overall")
    lines.append(f"")
    lines.append(f"- Complete: {len(complete)} / {len(results)}")
    lines.append(f"- Incomplete or missing: {len(incomplete)} / {len(results)}")
    if complete:
        lines.append(f"- OK: {', '.join(r['cls'] for r in complete)}")
    if incomplete:
        lines.append(f"- FAIL: {', '.join(r['cls'] for r in incomplete)}")
    lines.append(f"")

    # Summary table
    lines.append(f"## Per-Class Summary")
    lines.append(f"")
    lines.append(f"| Class | train/good | test/good | test defect | gt mask | Missing | Usable |")
    lines.append(f"|-------|-----------|-----------|-------------|---------|---------|--------|")
    for r in results:
        usable = "Yes" if not r["missing"] else "No"
        miss_str = ", ".join(r["missing"]) if r["missing"] else "-"
        lines.append(
            f"| {r['cls']} | {r['train_good']} | {r['test_good']} | {r['test_defect']} | {r['gt_mask']} | {miss_str} | {usable} |"
        )
    lines.append(f"")

    # Defect type details
    lines.append(f"## Defect Type Details")
    lines.append(f"")
    for r in results:
        lines.append(f"### {r['cls']}")
        if r["missing"]:
            lines.append(f"- **Missing**: {', '.join(r['missing'])}")
        if r["test_types"]:
            lines.append(f"- test defects: {', '.join(f'{n}({c})' for n, c in r['test_types'])}")
        if r["gt_types"]:
            lines.append(f"- ground_truth masks: {', '.join(f'{n}({c})' for n, c in r['gt_types'])}")
        if not r["test_types"] and not r["gt_types"] and not r["missing"]:
            lines.append(f"- (no defect data)")
        lines.append(f"")

    # Recommendations
    if complete:
        lines.append(f"## Recommended Classes to Run First")
        lines.append(f"")
        recs = []
        for r in complete:
            if r["train_good"] > 0 and r["test_defect"] > 0:
                recs.append(r["cls"])
        if recs:
            for c in recs[:3]:
                lines.append(f"- `{c}`")
        lines.append(f"")

    return "\n".join(lines)


def main():
    results = [check_class(DATA, c) for c in TARGET]
    md = generate_report(results)
    print(md)
    REPORT.write_text(md, encoding="utf-8")
    print(f"\nReport saved to: {REPORT}")


if __name__ == "__main__":
    main()
