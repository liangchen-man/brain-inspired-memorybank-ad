#!/usr/bin/env python3
"""Week 2: Workstation anomaly monitoring video audit.

Reads 视频数据集.zip, extracts mp4 videos to data/workstation/raw/,
reads metadata, extracts audit frames at low FPS, generates contact sheets,
and produces CSV + Markdown reports.

Does NOT: run anomaly detection, train models, delete the zip, or label data.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path("E:/leinaozuoye")
ZIP_PATH = Path("E:/下载/视频数据集.zip")
RAW_DIR = PROJECT_ROOT / "data" / "workstation" / "raw"
AUDIT_DIR = PROJECT_ROOT / "data" / "workstation" / "audit_frames"
MANIFEST_DIR = PROJECT_ROOT / "data" / "workstation" / "manifests"
CONTACT_DIR = PROJECT_ROOT / "report" / "figures" / "week2_audit_contact_sheet"
RESULTS_DIR = PROJECT_ROOT / "results" / "week2"
REPORT_PATH = PROJECT_ROOT / "report" / "week2_video_audit.md"
ERROR_LOG_PATH = PROJECT_ROOT / "ERROR_LOG.md"

AUDIT_FPS = 0.5
AUDIT_MAX_FRAMES = 300
CONTACT_COLS = 6
CONTACT_THUMB_W = 200


def log_error(issue_num: int, title: str, cmd: str, err: str, cause: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n## Issue {issue_num} — {title}\n"
        f"- Time: {now}\n"
        f"- Command: {cmd}\n"
        f"- Error: {err}\n"
        f"- Suspected cause: {cause}\n"
        f"- Status: open\n"
    )
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


def get_error_count() -> int:
    """Rough count of existing issues."""
    if not ERROR_LOG_PATH.exists():
        return 0
    text = ERROR_LOG_PATH.read_text(encoding="utf-8")
    count = 0
    for line in text.splitlines():
        if line.startswith("## Issue "):
            count += 1
    return count


def fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n/1_000:.2f} KB"
    return f"{n} B"


def zip_inventory() -> list[dict]:
    """List all files in the zip."""
    items = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            items.append({
                "filename": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "is_dir": info.is_dir(),
            })
    return items


def extract_mp4s(zip_items: list[dict]) -> list[Path]:
    """Extract only .mp4 files to RAW_DIR (flattened). Return list of extracted paths."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extracted = []
    mp4_items = [i for i in zip_items if i["filename"].lower().endswith(".mp4")]
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for item in mp4_items:
            basename = Path(item["filename"]).name
            dest = RAW_DIR / basename
            if not dest.exists():
                print(f"Extracting: {item['filename']} -> {dest}")
                # Use a temp extract then move, since z.extract preserves dirs
                z.extract(item["filename"], RAW_DIR)
                nested = RAW_DIR / item["filename"]
                if nested.exists() and nested != dest:
                    import shutil
                    # Move via copy since on Windows rename across drives can fail
                    shutil.move(str(nested), str(dest))
                    # Clean up empty parent dirs
                    parent = nested.parent
                    while parent != RAW_DIR:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent
            else:
                print(f"Already exists, skip: {dest}")
            extracted.append(dest)
    return extracted


def read_video_meta(video_path: Path) -> dict:
    """Read video metadata with OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "file_size_bytes": video_path.stat().st_size,
            "duration_seconds": None,
            "fps": None,
            "width": None,
            "height": None,
            "frame_count": None,
            "codec": None,
            "error": "cv2.VideoCapture failed to open",
        }

    meta = {
        "video_path": str(video_path),
        "file_size_bytes": video_path.stat().st_size,
        "duration_seconds": None,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "error": None,
    }

    # Decode FOURCC
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    if fourcc_int > 0:
        fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
        meta["codec"] = fourcc_str

    # Duration from frame_count / fps
    if meta["fps"] and meta["fps"] > 0 and meta["frame_count"]:
        meta["duration_seconds"] = round(meta["frame_count"] / meta["fps"], 2)

    cap.release()
    return meta


def extract_audit_frames(
    video_path: Path,
    fps: float,
    meta: dict,
    max_frames: int,
    case_id: str,
) -> list[dict]:
    """Extract frames at given FPS, cap at max_frames."""
    video_dir = AUDIT_DIR / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = meta.get("fps") or 30.0
    total_frames = meta.get("frame_count") or 0
    duration = meta.get("duration_seconds") or 0

    # Calculate interval: every N source frames
    if fps > 0 and src_fps > 0:
        interval = max(1, int(src_fps / fps))
    else:
        interval = 60  # fallback

    # Estimate how many frames we'd get; if > max_frames, use a lower effective FPS
    estimated = int(total_frames / interval) if total_frames > 0 else 0
    effective_fps = fps
    if estimated > max_frames:
        # Increase interval to stay under max_frames
        interval = max(1, int(total_frames / max_frames))
        effective_fps = src_fps / interval

    records = []
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0 and saved < max_frames:
            timestamp = frame_idx / src_fps if src_fps > 0 else 0.0
            frame_name = f"{video_path.stem}_{saved:06d}_{timestamp:.2f}s.png"
            frame_path = video_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            h, w = frame.shape[:2]
            records.append({
                "case_id": case_id,
                "video_id": video_path.stem,
                "frame_path": str(frame_path),
                "timestamp_seconds": round(timestamp, 3),
                "frame_index": frame_idx,
                "width": w,
                "height": h,
                "source_video": str(video_path),
            })
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  Extracted {saved} frames @ ~{effective_fps:.2f} FPS from {video_path.name}")
    return records


def make_contact_sheet(
    frame_records: list[dict],
    video_stem: str,
    cols: int = CONTACT_COLS,
    thumb_w: int = CONTACT_THUMB_W,
) -> Path | None:
    """Generate a contact sheet PNG from extracted frames."""
    if not frame_records:
        return None

    paths = [Path(r["frame_path"]) for r in frame_records if Path(r["frame_path"]).exists()]
    if not paths:
        return None

    # Evenly sample up to cols * 8 ≈ 48 frames for the contact sheet
    max_tiles = cols * 8
    if len(paths) > max_tiles:
        step = len(paths) // max_tiles
        paths = paths[::step][:max_tiles]

    # Read and resize thumbnails
    thumbs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = int(h * thumb_w / w)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        thumbs.append(thumb)

    if not thumbs:
        return None

    # Arrange into grid
    rows = (len(thumbs) + cols - 1) // cols
    # Use the max height among thumbs in each row
    row_imgs = []
    for r in range(rows):
        row_thumbs = thumbs[r * cols : (r + 1) * cols]
        # Pad to uniform height in this row
        row_h = max(t.shape[0] for t in row_thumbs)
        padded = []
        for t in row_thumbs:
            if t.shape[0] < row_h:
                pad = np.zeros((row_h - t.shape[0], thumb_w, 3), dtype=np.uint8)
                t = np.vstack([t, pad])
            padded.append(t)
        # Pad to full width if last row is short
        while len(padded) < cols:
            padded.append(np.zeros((row_h, thumb_w, 3), dtype=np.uint8))
        row_imgs.append(np.hstack(padded))

    contact = np.vstack(row_imgs)

    # Add timestamp labels
    out_path = CONTACT_DIR / f"{video_stem}_contact_sheet.png"
    cv2.imwrite(str(out_path), contact)
    print(f"  Contact sheet saved: {out_path} ({len(paths)} tiles)")
    return out_path


def process_video(
    video_path: Path,
    case_id: str,
) -> dict | None:
    """Full pipeline for one video: meta -> audit frames -> contact sheet."""
    print(f"\nProcessing: {video_path.name}")

    meta = read_video_meta(video_path)
    if meta["error"]:
        print(f"  ERROR reading metadata: {meta['error']}")
        return None

    print(f"  Resolution: {meta['width']}x{meta['height']}, "
          f"FPS: {meta['fps']:.2f}, "
          f"Frames: {meta['frame_count']}, "
          f"Duration: {meta['duration_seconds']}s, "
          f"Codec: {meta['codec']}")

    # Determine effective FPS
    use_fps = AUDIT_FPS
    est = int(meta["frame_count"] / (meta["fps"] / AUDIT_FPS)) if meta["fps"] and meta["fps"] > 0 else 0
    if est > AUDIT_MAX_FRAMES:
        use_fps = AUDIT_MAX_FRAMES / (meta["frame_count"] / meta["fps"]) if meta["fps"] and meta["frame_count"] else 0.2

    frame_records = extract_audit_frames(
        video_path=video_path,
        fps=use_fps,
        meta=meta,
        max_frames=AUDIT_MAX_FRAMES,
        case_id=case_id,
    )

    contact_path = make_contact_sheet(frame_records, video_path.stem)

    return {
        "meta": meta,
        "frame_records": frame_records,
        "contact_path": contact_path,
        "effective_fps": use_fps,
    }


def write_inventory_csv(videos_meta: list[dict]) -> None:
    """Write video_inventory.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "video_inventory.csv"
    headers = [
        "case_id", "video_path", "file_size_bytes", "duration_seconds",
        "fps", "width", "height", "frame_count", "codec", "notes",
    ]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for m in videos_meta:
            w.writerow({
                "case_id": m.get("case_id", "workstation"),
                "video_path": m.get("video_path", ""),
                "file_size_bytes": m.get("file_size_bytes", "null"),
                "duration_seconds": m.get("duration_seconds", "null"),
                "fps": m.get("fps", "null"),
                "width": m.get("width", "null"),
                "height": m.get("height", "null"),
                "frame_count": m.get("frame_count", "null"),
                "codec": m.get("codec", "null"),
                "notes": m.get("error") or "",
            })
    print(f"\nInventory CSV: {out}")


def write_audit_frames_csv(all_frames: list[dict]) -> None:
    """Write audit_frames.csv manifest."""
    out = MANIFEST_DIR / "audit_frames.csv"
    headers = [
        "case_id", "video_id", "frame_path", "timestamp_seconds",
        "frame_index", "width", "height", "source_video",
    ]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in all_frames:
            w.writerow(r)
    print(f"Audit frames manifest: {out} ({len(all_frames)} rows)")


def write_markdown_report(
    zip_items: list[dict],
    video_results: list[dict],
    all_frame_records: list[dict],
) -> None:
    """Write week2_video_audit.md."""
    lines = [
        "# Week 2: Workstation Video Audit Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source**: `E:/下载/视频数据集.zip`",
        "",
        "---",
        "",
        "## 1. ZIP Internal File List",
        "",
    ]

    for item in zip_items:
        tag = "[DIR]" if item["is_dir"] else ""
        lines.append(f"- {item['filename']}  {tag}  {fmt_bytes(item['file_size'])}")

    lines.extend([
        "",
        "## 2. Video Metadata",
        "",
        "| Property | Video 1 | Video 2 |",
        "|----------|---------|---------|",
    ])

    if len(video_results) >= 2:
        r0 = video_results[0]["meta"]
        r1 = video_results[1]["meta"]
        rows = [
            ("Filename", Path(r0["video_path"]).name, Path(r1["video_path"]).name),
            ("File size", fmt_bytes(r0["file_size_bytes"]), fmt_bytes(r1["file_size_bytes"])),
            ("Duration (s)", str(r0["duration_seconds"]), str(r1["duration_seconds"])),
            ("FPS", str(r0["fps"]), str(r1["fps"])),
            ("Resolution", f"{r0['width']}x{r0['height']}", f"{r1['width']}x{r1['height']}"),
            ("Frame count", str(r0["frame_count"]), str(r1["frame_count"])),
            ("Codec", str(r0.get("codec", "null")), str(r1.get("codec", "null"))),
        ]
        for label, v1, v2 in rows:
            lines.append(f"| {label} | {v1} | {v2} |")
    elif video_results:
        r0 = video_results[0]["meta"]
        lines.append(f"| Filename | {Path(r0['video_path']).name} | — |")
        lines.append(f"| File size | {fmt_bytes(r0['file_size_bytes'])} | — |")
        lines.append(f"| Duration (s) | {r0['duration_seconds']} | — |")
        lines.append(f"| FPS | {r0['fps']} | — |")
        lines.append(f"| Resolution | {r0['width']}x{r0['height']} | — |")
        lines.append(f"| Frame count | {r0['frame_count']} | — |")
        lines.append(f"| Codec | {r0.get('codec', 'null')} | — |")

    lines.extend([
        "",
        "## 3. Audit Frame Extraction Strategy",
        "",
        f"- Target FPS: {AUDIT_FPS}",
        f"- Max frames per video: {AUDIT_MAX_FRAMES}",
        f"- If estimated frames exceed max, FPS is lowered to stay within the limit.",
        "",
        "| Video | Effective FPS | Frames Extracted |",
        "|-------|--------------|-----------------|",
    ])
    for vr in video_results:
        stem = Path(vr["meta"]["video_path"]).stem
        lines.append(f"| {stem} | {vr['effective_fps']:.3f} | {len(vr['frame_records'])} |")

    lines.extend([
        "",
        "## 4. Contact Sheets",
        "",
    ])
    for vr in video_results:
        stem = Path(vr["meta"]["video_path"]).stem
        cp = vr.get("contact_path")
        if cp:
            lines.append(f"- [{stem} contact sheet](figures/week2_audit_contact_sheet/{cp.name})")
        else:
            lines.append(f"- {stem}: FAILED to generate")

    lines.extend([
        "",
        "## 5. Preliminary Visual Observations",
        "",
        "> **Note**: These are executor-level observations from contact sheets and sample frames only.",
        "> No pixel-level analysis has been performed. Codex / user must confirm all conclusions.",
        "",
        "### Common observations (both videos)",
        "",
        "| Observation | Judgment |",
        "|------------|----------|",
        "| Fixed camera | 需要 Codex / 用户确认 — contact sheet shows similar viewpoint but human review needed |",
        "| Looks like workstation monitoring | 需要 Codex / 用户确认 |",
        "| Obvious ROI present | 需要 Codex / 用户确认 — possibly desk/monitor area |",
        "| Human hand / tool occlusion | 需要 Codex / 用户确认 |",
        "| Stage changes visible | 需要 Codex / 用户确认 |",
        "| Suspicious anomaly visible | 需要 Codex / 用户确认 — contact sheet alone insufficient |",
        "",
        "### Per-video notes",
        "",
    ])
    for vr in video_results:
        stem = Path(vr["meta"]["video_path"]).stem
        dur = vr["meta"]["duration_seconds"]
        lines.append(f"#### {stem}")
        if dur and dur > 3600:
            lines.append(f"- Long video ({dur/3600:.1f} hours) — may contain multiple activity phases.")
        lines.append(f"- {len(vr['frame_records'])} audit frames extracted.")
        lines.append("- 需要 Codex 通过 contact sheet 或逐帧 Review 确认内容。")
        lines.append("")

    # Write
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")


def main() -> int:
    error_base = get_error_count()
    errors = []

    # 1. Make dirs
    for d in [RAW_DIR, AUDIT_DIR, MANIFEST_DIR, CONTACT_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. ZIP inventory
    print("=== Step 1: ZIP inventory ===")
    zip_items = zip_inventory()
    for item in zip_items:
        tag = "[DIR]" if item["is_dir"] else ""
        print(f"  {item['filename']:60s} {tag:6s} {fmt_bytes(item['file_size'])}")

    # 3. Extract mp4s
    print("\n=== Step 2: Extract MP4 files ===")
    extracted = extract_mp4s(zip_items)
    print(f"  Extracted {len(extracted)} video(s)")

    # 4. Process each video
    print("\n=== Step 3: Process videos ===")
    video_results = []
    all_frame_records = []
    for i, vpath in enumerate(extracted):
        case_id = f"workstation_video_{i+1}"
        try:
            result = process_video(vpath, case_id)
            if result:
                result["meta"]["case_id"] = case_id
                video_results.append(result)
                all_frame_records.extend(result["frame_records"])
        except Exception as exc:
            msg = f"Failed to process {vpath.name}: {exc}"
            print(f"  ERROR: {msg}")
            errors.append(msg)

    # 5. Write CSVs
    print("\n=== Step 4: Write CSVs ===")
    write_inventory_csv([vr["meta"] for vr in video_results])
    write_audit_frames_csv(all_frame_records)

    # 6. Write report
    print("\n=== Step 5: Write report ===")
    write_markdown_report(zip_items, video_results, all_frame_records)

    # 7. Log errors
    if errors:
        for i, err in enumerate(errors):
            log_error(
                issue_num=error_base + i + 1,
                title="Week2 video audit error",
                cmd="python src/week2_video_audit.py",
                err=err,
                cause="Video processing failure",
            )
        print(f"\n{len(errors)} error(s) logged to ERROR_LOG.md")

    print("\n=== Done ===")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
