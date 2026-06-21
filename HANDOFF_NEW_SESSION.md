# HANDOFF — Final Report v2 Draft Phase

**From**: Codex / Claude Code
**To**: Codex / User
**Date**: 2026-06-21
**Updated**: 2026-06-21 (paper_claude_package_v2 generated; GitHub repo initialized, awaiting remote URL)

---

## Executive Summary

No new experiments should be run. The project has moved from experiment execution to final report review/formatting. Codex created the main Chinese final report draft:

```text
report/final_report_v2.md
```

This draft integrates Week 1 MVTec AD results, two ablations, Week 2 Case 4 three-method acquisition/observability failure, Week 2 Case 2 temporal baseline with HUMAN_INTERACTION and `4→3→4→3`, brain-inspired memory discussion, frontier-method comparison, deployment suggestions, limitations, and the current GPU rental decision.

All prior experiment deliverables (Week 1 MVTec AD, Week 2 Case 4 three-method convergence, Week 2 Case 2 temporal state baseline v2) remain intact and unmodified.

---

## Current Main Deliverable

| File | Description |
|------|-------------|
| `report/final_report_v2.md` | Main Chinese report draft, 295 lines, about 21 KB. This is now the primary file to review, polish, format, and eventually convert to Word/PDF if required. |

## Supporting Deliverables

| File | Description |
|------|-------------|
| `report/frontier_methods_comparison.md` | 7-method comparison table (DMAD, SimpleNet, Dinomaly, AnomalyGPT, Diffusion, EfficientAD, 本项目); 5-reason explanation of why SOTA not implemented; brain-inspired integration into final report |
| `report/final_report_v2_outline.md` | 12-chapter outline for final Chinese report — each chapter with 3-6 bullet points ready for expansion |
| `report/final_report_integration_checklist.md` | Comprehensive checklist: mandatory citations (12 files), mandatory figures (13 items, 8 new/missing), mandatory tables (13 items), course questions (7 items), "do not do" list, Codex decision questions, chapter-by-chapter integration status matrix |

## P0 Figure Assets (Session 2 — 2026-06-21)

| File | Description |
|------|-------------|
| `report/final_assets/fig_week1_four_algorithms_comparison.png` | 1×5 layout comparison (Original/PaDiM/PatchCore/SPADE/AnomalyDINO) — 150dpi, ~3.5 MB, CJK labels via Microsoft YaHei |
| `report/final_assets/table_case4_three_method_failure.md` | 4 comprehensive tables: v1/v2/v3 config+results, v1 vs v2 rank change, v3 z-score matrix, root cause diagnosis |
| `report/final_assets/fig_static_to_dynamic_memory.md` | Mermaid flowchart: MVTec AD static bank → Case 4 acquisition gate → Case 2 temporal state → 3-layer brain-inspired memory framework |
| `report/final_report_assets_index.md` | Complete asset index: all P0/P1 figures, existing reusable materials, table checklist, completion status, Codex review items |
| `tools/gen_fig_week1_comparison.py` | Idempotent regeneration script for P0-1 (reads overlays from runs/ subdirectories) |

---

## Project Phase

**Current phase**: Final report review and formatting.

Recommended next tasks:
1. Review `report/final_report_v2.md` for course requirement coverage and claim safety.
2. Decide whether to make optional P1 figures (human intervention contact sheet, brain-inspired framework diagram, five-level Case 2 argument figure).
3. Render `report/final_assets/fig_static_to_dynamic_memory.md` from Mermaid to PNG if Word/PDF output is needed.
4. Convert final Markdown to Word/PDF only after technical review.

---

## What NOT to Do (Reconfirmed)
- Do NOT run any new experiments
- Do NOT run PatchCore/DINOv2/deep models
- Do NOT modify `src/` scripts
- Do NOT modify existing `results/` or `runs/`
- Do NOT rent GPU
- Do NOT implement DMAD / Dinomaly / diffusion models
- Do NOT fabricate literature metrics
- Do NOT claim unimplemented methods as implemented

---

## Key Arguments (Paper-Ready, from prior sessions)

1. **Why static memory bank fails on Case 4**: signal below noise floor — "采集方案比算法更重要"
2. **Why single-frame detection fails on Case 2**: 5-level argument (pixel instability, dual semantics, human hand = false positive, normality = temporal set, state machine = free signal)
3. **Why short-term memory is needed**: distinguish "normal empty" from "abnormal missing", suppress motion blur false positives, detect human intervention as "needs confirmation" not "anomaly"
4. **Why long-term memory matters**: store cycle periods, human intervention frequency, 4⇄3 reversibility as "known normal pattern", drift detection
5. **Brain-inspired 3-layer memory framework**: long-term memory (static bank) + short-term memory (sliding window) + feedback gating (human confirmation)

---

## Questions for Codex/User

- [ ] **Is 4→3→4→3 standard operating procedure?** (unload → inspect/reload → unload again)
- [ ] **t≈380-389s 1st removal**: automated ejection or same human (missed by 2s sampling)?
- [ ] **t≈403-405s 2nd removal**: automated or human?
- [ ] **Human intervention frequency**: once per batch normal, or this video is special?
- [ ] **Brightness transition zone root cause**: camera auto-exposure, lighting change, or equipment action?
- [ ] **Report language**: Chinese, English, or bilingual?
- [ ] **Citation format**: IEEE, GB/T 7714, or other?
- [ ] **Frontier methods chapter depth**: detailed discussion or brief list?
- [ ] **Heatmap layout**: per-algorithm 3-5 representative images, or unified multi-panel comparison?
- [ ] **Figure production order**: confirmation needed for P0 figures (4-algorithm heatmap multi-panel, Case 4 three-method comparison table, static→dynamic memory flowchart)

---

## Figure Production Status

**P0 — Completed**:
1. Week 1 four-algorithm heatmap comparison (multi-panel)
2. Case 4 three-method failure comparison table
3. Static memory bank → dynamic memory flowchart

**P1 — Completed**:
4. ✅ Case 2 human intervention contact sheet (6 frames with red borders)
5. ✅ Brain-inspired 3-layer memory framework diagram
6. ✅ Case 2 five-level argument illustration
7. ✅ Static memory bank -> dynamic memory flowchart rendered as PNG

**P2 — Nice to have (not done)**:
8. Case 4 sub-ROI annotation on video frame
9. Case 2 4➜3➜4➜3 event sequence illustration
6. Case 2 five-level argument illustration

**P2 — Nice to have**:
7. Case 4 sub-ROI annotation on video frame
8. Case 2 4→3→4→3 event sequence illustration

---

## 2026-06-21 Update: Packaging & Git Phase

### paper_claude_package_20260621_v2.zip
- Path: `E:\leinaozuoye\paper_claude_package_20260621_v2.zip`
- Size: 6.9 MB, 51 entries
- 42 source files, zero missing
- `README_FOR_CLAUDE.md` written in Chinese with full writing constraints
- Zip integrity verified, no large files, no raw data, no videos

### Git Repository
- `git init` completed, 98 files committed on `master`
- `.gitignore` configured to exclude: data/, runs/, memory/, *.pt, *.zip, *.mp4, *.pdf, MVTecAD1/, VisA/
- README.md rewritten with professional project overview
- REPORT_INDEX.md created
- **No remote configured** — awaiting GitHub repo URL from user
- Push pending: once remote is added, run `git remote add origin <URL> && git push -u origin master`
