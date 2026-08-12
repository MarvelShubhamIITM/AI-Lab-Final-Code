---
title: Driver Wellness AI
emoji: 🚗
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Driver Wellness AI — Integrated Inference

End-to-end driver monitoring: five models (video fatigue, landmark fatigue,
driver activity, seat-belt/phone, smoking/drinking), risk fusion, and an
annotated dashboard. Two tabs:

- **📹 Recorded video** — upload a short driving clip to get a fused, annotated
  wellness/risk analysis.
- **🔴 Live webcam** — stream your webcam for a real-time fused analysis. The
  browser streams frames to the same streaming orchestrator used for recorded
  video, so the annotated feed and risk score update continuously.

## Models
| Module | File |
|--------|------|
| Video Fatigue (EfficientNet-B0 + BiLSTM) | `models/Video_Fatigue.pth` |
| Landmark Fatigue (MediaPipe + LSTM) | `models/Landmark_Fatigue.pt` |
| Driver Activity (MobileNetV3-Large) | `models/Driver_Activity.pth` |
| Smoking & Drinking (YOLOv8n) | `models/Smoking_And_Drinking.pt` |
| Seat Belt & Phone (YOLOv8n) | `models/SeatBelt_And_Phone.pt` |

Landmark fatigue also needs `models/m4_normalization_stats_ws45.csv` and
`models/face_landmarker.task` (the MediaPipe asset is auto-downloaded on first
use if missing).

## Risk fusion
Scores are combined with the **Common Driver Risk Score Framework (Option A)**:

```
R_i          = severity_w_i × confidence   (only for risky predictions; safe -> 0)
R_total      = Σ R_i  over ready modules
overall_score = 100 × (1 − exp(−k × R_total))          (k = 0.05)
```

The fused score is optionally smoothed with a short moving-average window during
streaming (PDF Step 5).

## Notes
- Runs on CPU by default (slow). Upgrade Space hardware to a GPU for real-time
  speed — this matters most for the **Live webcam** tab.
- The live tab replaces the notebook's Colab-JS / local-OpenCV webcam loops with
  Gradio's browser webcam streaming, which is how live capture works inside a
  Hugging Face Space.

> The `sdk_version` above should match a Gradio release that exists at deploy
> time — bump it if the build log complains.
