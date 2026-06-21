# SPADE Preflight Check

**Checked**: 2026-06-14  
**Checked by**: DeepSeek read-only inspection, then updated by Codex after creating `src/run_spade.py`

---

## Verdict: READY FOR TEACHING SPADE CARPET

The official SPADE repository is still unavailable because `external/` is empty. However, Codex has created a teaching SPADE-style runner:

```text
src/run_spade.py
```

This is enough for a minimal course experiment on `carpet`. It must be reported as a teaching implementation, not an official SPADE reproduction.

---

## Precondition Summary

| # | Precondition | Status | Detail |
|---|---|---|---|
| 1 | `external/spade-pytorch/` exists | WARN | Not found; official repo unavailable |
| 2 | `src/run_spade.py` exists | OK | Created by Codex |
| 3 | `src/run_spade.py` syntax check | OK | `python -m py_compile src/run_spade.py` passed |
| 4 | `src/run_spade.py --help` | OK | CLI arguments are visible |
| 5 | `data/mvtec/carpet/` complete | OK | train/good 280; test and ground_truth available |
| 6 | `braincv-ad-py310` env ready | OK | Python 3.10.20; PyTorch CUDA available |
| 7 | Existing PaDiM/PatchCore results | OK | Four completed results are intact |

---

## Teaching Route

Current route:

```text
Teaching SPADE-style baseline
ResNet18 layer2 features
normal image global feature memory
nearest normal image retrieval
local spatial feature discrepancy heatmap
no external repo dependency
```

This route is appropriate for the report section on normality memory because it emphasizes the idea of retrieving similar normal references before localizing abnormal deviations.

---

## Next Command

Run only carpet first:

```powershell
conda activate braincv-ad-py310
python src/run_spade.py --category carpet --data-root data/mvtec --output runs/spade_carpet --device cuda --image-size 224 --batch-size 4 --feature-dim 64 --k-neighbors 5 --max-heatmaps 30
```

Do not run bottle or AnomalyDINO until Codex reviews the carpet result.
