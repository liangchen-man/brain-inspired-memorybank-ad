# Literature and Method Notes

更新时间：2026-06-14

## AnomalyDINO

- Source: AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2  
  URL: https://arxiv.org/abs/2405.14529
- Key point for this project: AnomalyDINO adapts DINOv2 for one-shot and few-shot anomaly detection. The method is training-free, uses patch similarities, and supports both image-level anomaly prediction and pixel-level anomaly segmentation.
- How it informs our implementation: our teaching runner should use DINO/ViT patch tokens as a normal patch memory bank, then score test patch tokens by nearest-neighbor distance to normal tokens.
- Limitation: the project does not currently clone an official AnomalyDINO repository, so the implementation must be reported as `AnomalyDINO-teaching`, not official reproduction.

## DINOv2

- Source: facebookresearch/dinov2  
  URL: https://github.com/facebookresearch/dinov2
- Key point for this project: the official repository describes DINOv2 as self-supervised visual features that can be used across computer vision tasks without fine-tuning. It also provides ViT-S/14, ViT-B/14, ViT-L/14 and ViT-g/14 backbones.
- How it informs our implementation: DINOv2 patch tokens are a reasonable representation for a normality memory bank, especially when the report discusses memory-based anomaly detection rather than supervised defect classification.

## timm / pytorch-image-models

- Source: huggingface/pytorch-image-models  
  URL: https://github.com/huggingface/pytorch-image-models
- Key point for this project: `timm` provides many PyTorch image backbones, including ViT/DINO-family model entries available in the local environment.
- Local observation: `vit_small_patch14_dinov2` defaults to a 518 input size in the current `timm` configuration, but `img_size=224` works for extracting a 16x16 token grid in our preflight probe.

## Reporting Rule

Do not write that the project reproduced official AnomalyDINO unless an official repository is actually cloned and executed. The current route is a teaching implementation inspired by the paper: DINO/ViT patch-token memory, coreset compression, and nearest-neighbor anomaly scoring.
