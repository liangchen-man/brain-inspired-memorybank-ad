# PaDiM Results Summary

**Generated**: 2026-06-14
**Source**: `runs/padim_carpet/results.json`, `runs/padim_bottle/results.json`

---

## Results Table

| Property | carpet | bottle |
|----------|--------|--------|
| Status | success | success |
| Algorithm | PaDiM-teaching | PaDiM-teaching |
| Backbone | ResNet18 layer2 | ResNet18 layer2 |
| Image size | 224 | 224 |
| Feature dim | 64 | 64 |
| Train images | 280 | 209 |
| Test images | 117 | 83 |
| Image AUROC | 0.99358 | 0.99603 |
| Pixel AUROC | 0.98616 | 0.98004 |
| Runtime (seconds) | 231.634 | 69.473 |
| Max GPU memory (MB) | 31.936 | 31.936 |
| Memory bank file bytes | 13,049,029 | 13,049,029 |
| Heatmaps saved | 30 | 30 |
| Overlays saved | 30 | 30 |
| Error | null | null |

## Defect Types

**carpet**: color(19) + cut(11) + hole + metal_contamination + thread

**bottle**: broken_large(20) + broken_small(10) + contamination

## Files

```
runs/padim_carpet/
  results.json    OK
  run_log.md      OK
  memory_bank.pt  OK (12.4 MB)
  heatmaps/       30 PNGs
  overlays/       30 PNGs

runs/padim_bottle/
  results.json    OK
  run_log.md      OK
  memory_bank.pt  OK (12.4 MB)
  heatmaps/       30 PNGs
  overlays/       30 PNGs
```

## Discrepancy Note

bottle `runtime_seconds` in results.json is **69.473** but the previously generated run_log.md shows **108.128**. The second run (triggered by the preflight check's `conda run` command) re-executed the experiment and overwrote the original results. The current results.json value (69.473) is the authoritative latest value.
