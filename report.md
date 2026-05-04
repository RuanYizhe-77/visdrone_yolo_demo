# VisDrone Demo Report

## Goal

Build a GitHub-ready VisDrone detection and tracking demo with a custom low-level PyTorch detector, an older baseline, and a correct VisDrone-pretrained YOLO reference.

## Final Comparison

| model | role | tracker | precision | recall | mAP50 approx | boxes/image |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pretrained YOLO | external VisDrone reference | ByteTrack-style | 0.6971 | 0.4957 | 0.2960 | 50.30 |
| newerbest P2 custom | improved custom model | ByteTrack-style | 0.4041 | 0.4796 | 0.2211 | 83.94 |
| old 50-epoch custom | baseline | simple IoU | 0.2823 | 0.2504 | 0.0593 | 62.74 |

The simplified validation ranking is:

```text
pretrained VisDrone YOLO > newer P2 custom > old 50-epoch custom
```

## Key Assets

- Main report: `assets/current_yolo_newerbest_oldest/comparison_report.md`
- Score table: `assets/current_yolo_newerbest_oldest/scores/metrics_summary.csv`
- Image grids: `assets/current_yolo_newerbest_oldest/images/`
- Clear tracking video 1: `assets/presentation_tracking_comparison/tracking/uav0000268_05773_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
- Clear tracking video 2: `assets/presentation_tracking_comparison_uav0000268_late/tracking/uav0000268_05773_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
- Additional tracking video: `assets/presentation_tracking_comparison_uav0000305/tracking/uav0000305_00000_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
- Recommended stricter ByteTrack video: `assets/presentation_tracking_comparison_uav0000305_highthr/tracking/uav0000305_00000_v_pretrained_yolo_vs_newerbest_p2_small_vs_oldest_50epoch.mp4`
- On `uav0000305_00000_v`, the stricter thresholds reduced newer-model created IDs from 1590 to 309 across the same 184 frames.

## What Changed

- Added P2 stride-4 feature support for small VisDrone objects.
- Added stride-normalized box regression with Softplus activation.
- Added classwise-then-agnostic NMS for duplicate suppression.
- Added ByteTrack-style tracking and a stable demo variant.
- Added a stricter ByteTrack render for the clearest `uav0000305_00000_v` comparison.
- Added a final comparison suite that generates image grids, tracking videos, score tables, and reports.
- Improved comparison video titles with high-contrast black title bars and yellow text.

## Limitations

- Metrics are simplified AP50-style metrics, not official VisDrone or COCO scores.
- The custom detector is educational and still weaker than the pretrained YOLO reference.
- ByteTrack-style tracking improves continuity, but it cannot fix missed or incorrect detections.
- Generated assets, datasets, and checkpoints are local artifacts and are ignored by Git by default.
