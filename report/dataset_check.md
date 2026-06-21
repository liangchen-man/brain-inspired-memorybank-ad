# MVTec AD Dataset Check Report

- **Check time**: 2026-06-14 10:31:04
- **Data root**: `E:\leinaozuoye\data\mvtec`
- **Data root exists**: True

## Overall

- Complete: 2 / 8
- Incomplete or missing: 6 / 8
- OK: carpet, bottle
- FAIL: grid, leather, tile, wood, capsule, hazelnut

## Per-Class Summary

| Class | train/good | test/good | test defect | gt mask | Missing | Usable |
|-------|-----------|-----------|-------------|---------|---------|--------|
| carpet | 280 | 28 | 89 | 89 | - | Yes |
| grid | 0 | 0 | 0 | 0 | class dir | No |
| leather | 0 | 0 | 0 | 0 | class dir | No |
| tile | 0 | 0 | 0 | 0 | class dir | No |
| wood | 0 | 0 | 0 | 0 | class dir | No |
| bottle | 209 | 20 | 63 | 63 | - | Yes |
| capsule | 0 | 0 | 0 | 0 | class dir | No |
| hazelnut | 0 | 0 | 0 | 0 | class dir | No |

## Defect Type Details

### carpet
- test defects: color(19), cut(17), hole(17), metal_contamination(17), thread(19)
- ground_truth masks: color(19), cut(17), hole(17), metal_contamination(17), thread(19)

### grid
- **Missing**: class dir

### leather
- **Missing**: class dir

### tile
- **Missing**: class dir

### wood
- **Missing**: class dir

### bottle
- test defects: broken_large(20), broken_small(22), contamination(21)
- ground_truth masks: broken_large(20), broken_small(22), contamination(21)

### capsule
- **Missing**: class dir

### hazelnut
- **Missing**: class dir

## Recommended Classes to Run First

- `carpet`
- `bottle`
