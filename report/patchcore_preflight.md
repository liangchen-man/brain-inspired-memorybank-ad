# PatchCore Preflight Check

**Checked**: 2026-06-14  
**Checked by**: Codex controller after creating `src/run_patchcore.py`

---

## Verdict: READY FOR TEACHING PATCHCORE CARPET

Official `external/patchcore-inspection` is still unavailable, and Faiss is still not installed. However, Codex has created a teaching PatchCore runner that does not depend on Faiss:

```text
src/run_patchcore.py
```

This is enough to run the first minimal PatchCore-style experiment on `carpet`.

---

## Precondition Checklist

| # | Precondition | Status | Detail |
|---|-------------|--------|--------|
| 1 | `data/mvtec/carpet/` complete | OK | train 280, test defects 89, gt masks 89 |
| 2 | `data/mvtec/bottle/` complete | OK | train 209, test defects 63, gt masks 63 |
| 3 | `braincv-ad-py310` env ready | OK | Python 3.10.20, PyTorch 2.12.0+cu126 |
| 4 | `sklearn` available | OK | 1.7.2 |
| 5 | `faiss` installed | WARN | NOT INSTALLED; not required by teaching script |
| 6 | `external/patchcore-inspection/` exists | WARN | external/ is empty; official repo not available |
| 7 | `src/run_patchcore.py` exists | OK | Created and syntax-checked |
| 8 | GPU available | OK | RTX 3050 4GB |

---

## Implementation Route

Current route:

```text
Teaching PatchCore
ResNet18 layer2 patch features
random coreset
nearest-neighbor patch distance with torch.cdist
no faiss dependency
```

This is not a full official PatchCore reproduction. It is intended to provide a runnable, explainable memory-bank baseline for the course report.

---

## Next Command

Run only carpet first:

```powershell
conda activate braincv-ad-py310
python src/run_patchcore.py --category carpet --data-root data/mvtec --output runs/patchcore_carpet --device cuda --image-size 224 --batch-size 4 --feature-dim 64 --coreset-size 2000 --coreset-method random --max-heatmaps 30
```

Do not run bottle until Codex reviews the carpet result.
