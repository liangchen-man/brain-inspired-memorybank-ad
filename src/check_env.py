#!/usr/bin/env python3
"""只读检查 Python / Conda / PyTorch / CUDA 环境。不安装、不修改任何包。"""

import os
import sys
import platform
import subprocess
import datetime
from pathlib import Path

ROOT = Path(r"E:\leinaozuoye")
ENV_MD = ROOT / "ENVIRONMENT.md"

# ---------- helpers ----------
def try_import(name, version_attr="__version__"):
    try:
        m = __import__(name)
        v = getattr(m, version_attr, None) if hasattr(m, version_attr) else "installed"
        return v
    except ImportError:
        return "NOT INSTALLED"

def nvidia_smi(field="name"):
    """Return GPU properties via nvidia-smi. Returns "N/A" on failure."""
    try:
        cp = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return "N/A"

# ---------- 1..25 checks ----------
R = {}
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 1
R["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"

# 2
R["python_version"] = sys.version.split()[0]

# 3
conda_ver = "NOT INSTALLED"
try:
    cp = subprocess.run(["conda", "--version"], capture_output=True, text=True, timeout=15)
    if cp.returncode == 0:
        conda_ver = cp.stdout.strip()
except FileNotFoundError:
    pass
except Exception as e:
    conda_ver = f"ERROR: {e}"
R["conda_version"] = conda_ver

# 4
R["python_exe"] = sys.executable

# 5
env_raw = os.environ.get("CONDA_DEFAULT_ENV", "")
R["conda_env"] = env_raw if env_raw else "(not in conda env or not launched via conda run)"

# 6, 7, 10, 11, 12, 13, 14  PyTorch block
torch_installed = False
try:
    import torch
    torch_installed = True
    R["torch_installed"] = "Yes"
    R["torch_version"] = torch.__version__
    R["cuda_available"] = "Yes" if torch.cuda.is_available() else "No"
    R["torch_cuda"] = torch.version.cuda if torch.version.cuda else "N/A"
    if torch.cuda.is_available():
        R["gpu_name"] = torch.cuda.get_device_name(0)
        R["gpu_count"] = str(torch.cuda.device_count())
        R["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    else:
        R["gpu_name"] = nvidia_smi("name")
        R["gpu_count"] = "1" if R["gpu_name"] != "N/A" else "0"
        R["gpu_memory"] = nvidia_smi("memory.total")
except ImportError:
    R["torch_installed"] = "No"
    R["torch_version"] = "NOT INSTALLED"
    R["cuda_available"] = "No"
    R["torch_cuda"] = "N/A"
    R["gpu_name"] = nvidia_smi("name")
    R["gpu_count"] = "1" if R["gpu_name"] != "N/A" else "0"
    R["gpu_memory"] = nvidia_smi("memory.total")

# 8, 9 torchvision
R["torchvision_installed"] = "Yes" if try_import("torchvision") != "NOT INSTALLED" else "No"
R["torchvision_version"] = try_import("torchvision")

# 15 opencv
R["opencv"] = try_import("cv2")

# 16 scikit-learn
R["sklearn"] = try_import("sklearn")

# 17 faiss
R["faiss"] = try_import("faiss")

# 18 timm
R["timm"] = try_import("timm")

# 19 numpy
R["numpy"] = try_import("numpy")

# 20 scipy
R["scipy"] = try_import("scipy")

# 21 matplotlib
R["matplotlib"] = try_import("matplotlib")

# 22 pandas
R["pandas"] = try_import("pandas")

# 23 project root
R["project_root"] = str(ROOT)

# 24 data/mvtec
mvtec = ROOT / "data" / "mvtec"
R["data_mvtec_exists"] = "Yes" if mvtec.is_dir() else "No"
cats = []
if mvtec.is_dir():
    for d in sorted(mvtec.iterdir()):
        if d.is_dir():
            cats.append(d.name)
R["mvtec_categories"] = ", ".join(cats) if cats else "(empty)"

# 25 runs/, results/, report/
for dname in ["runs", "results", "report"]:
    p = ROOT / dname
    R[f"dir_{dname}"] = "Yes" if p.is_dir() else "No"


# ---------- markdown ----------
def build_md(r: dict) -> str:
    # Pkg table
    pkgs = [
        ("Python",          r["python_version"]),
        ("Conda",           r["conda_version"]),
        ("Conda Env Name",  r["conda_env"]),
        ("PyTorch",         r["torch_version"]),
        ("torchvision",     r["torchvision_version"]),
        ("NumPy",           r["numpy"]),
        ("SciPy",           r["scipy"]),
        ("OpenCV (cv2)",    r["opencv"]),
        ("scikit-learn",    r["sklearn"]),
        ("Faiss",           r["faiss"]),
        ("timm",            r["timm"]),
        ("Matplotlib",      r["matplotlib"]),
        ("Pandas",          r["pandas"]),
    ]
    pkg_rows = "\n".join(f"| {n} | {v} |" for n, v in pkgs)

    missing = [n for n, v in pkgs if v == "NOT INSTALLED"]

    install_map = {
        "PyTorch":      "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
        "torchvision":  "pip install torchvision",
        "NumPy":        "pip install numpy",
        "SciPy":        "pip install scipy",
        "OpenCV (cv2)": "pip install opencv-python",
        "scikit-learn": "pip install scikit-learn",
        "Faiss":        "conda install -c pytorch faiss-cpu    (or faiss-gpu if GPU ready)",
        "timm":         "pip install timm",
        "Matplotlib":   "pip install matplotlib",
        "Pandas":       "pip install pandas",
    }
    inst_lines = [f"- `{install_map[m]}`" for m in missing if m in install_map]
    inst_section = "\n".join(inst_lines) if inst_lines else "(none — all key packages installed)"

    md = f"""# Environment Report

**Check time**: {now}
**Checked by**: src/check_env.py (read-only)

## OS

| Item | Value |
|------|-------|
| OS | {r['os']} |
| Python executable | `{r['python_exe']}` |
| Conda environment | {r['conda_env']} |

## Package Summary

| Package | Version |
|---------|---------|
{pkg_rows}

## CUDA / GPU

| Property | Value |
|----------|-------|
| PyTorch installed | {r['torch_installed']} |
| CUDA available (torch) | {r['cuda_available']} |
| torch.version.cuda | {r['torch_cuda']} |
| GPU name | {r['gpu_name']} |
| GPU count | {r['gpu_count']} |
| GPU memory | {r['gpu_memory']} |
| Suitable for DL experiments | {"Yes" if r["gpu_name"] != "N/A" else "No — no GPU detected"} |

## Project Directories

| Directory | Exists | Note |
|-----------|--------|------|
| `{r['project_root']}` | Yes | project root |
| `data/mvtec/` | {r['data_mvtec_exists']} | categories: {r['mvtec_categories']} |
| `runs/` | {r['dir_runs']} | |
| `results/` | {r['dir_results']} | |
| `report/` | {r['dir_report']} | |

## Missing Packages

{chr(10).join('- ' + m for m in missing) if missing else '(none)'}

## Suggested Install Commands (do NOT auto-install)

{inst_section}
"""
    return md


md = build_md(R)
print(md)

ENV_MD.write_text(md, encoding="utf-8")
print(f"\n[OK] Written to {ENV_MD}")
