# Preflight Check: PaDiM + carpet

**Check time**: 2026-06-14 10:31  
**Checked by**: Codex controller after extracting `carpet` and `bottle`

---

## Verdict: CANNOT RUN

PaDiM on carpet still cannot be run now.

The dataset prerequisite is now satisfied for `carpet`, but the runtime environment and PaDiM entry script are still missing.

---

## Precondition Checklist

| # | Precondition | Status | Detail |
|---|-------------|--------|--------|
| 1 | `data/mvtec/` exists | OK | Directory exists |
| 2 | `data/mvtec/carpet/` exists | OK | Extracted from `carpet.rar` |
| 3 | `data/mvtec/carpet/train/good/` exists | OK | 280 images |
| 4 | `data/mvtec/carpet/test/` exists | OK | good + defect folders |
| 5 | `data/mvtec/carpet/ground_truth/` exists | OK | 89 masks |
| 6 | PyTorch installed | FAIL | NOT INSTALLED |
| 7 | torchvision installed | FAIL | NOT INSTALLED |
| 8 | OpenCV installed | FAIL | NOT INSTALLED |
| 9 | timm installed | FAIL | NOT INSTALLED |
| 10 | Faiss installed | WARN | NOT INSTALLED; optional for PaDiM |
| 11 | sklearn installed | OK | 1.6.1 in previous check |
| 12 | `braincv-ad-py310` conda env exists | FAIL | NOT FOUND in previous conda env list |
| 13 | `src/run_padim.py` exists | FAIL | NOT FOUND |
| 14 | `external/` contains PaDiM code | FAIL | external/ is empty |
| 15 | `runs/` exists | OK | Directory exists |
| 16 | `results/` exists | OK | Directory exists |

---

## Dataset State

`report/dataset_check.md` confirms:

| Class | train/good | test/good | test defect | gt mask | Usable |
|-------|-----------:|----------:|------------:|--------:|--------|
| carpet | 280 | 28 | 89 | 89 | Yes |
| bottle | 209 | 20 | 63 | 63 | Yes |

---

## Still Missing

1. Conda environment `braincv-ad-py310`.
2. PyTorch and torchvision.
3. OpenCV.
4. timm.
5. PaDiM entry script `src/run_padim.py`.

---

## Next Action

Do not run PaDiM yet.

The next required step is environment setup and verification:

```powershell
conda create -n braincv-ad-py310 python=3.10 -y
conda activate braincv-ad-py310
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python numpy scipy scikit-learn pillow matplotlib pandas pyyaml tqdm timm
python src/check_env.py
```

After the environment is verified, Codex should implement or approve `src/run_padim.py`.

---

## Not Generated

- No heatmap was generated.
- No AUROC was computed.
- No `results.json` was created.
- No model was downloaded.
- No training or inference was run.
