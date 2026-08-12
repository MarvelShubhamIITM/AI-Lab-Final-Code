# Put model weights in this folder

`wellness_core.py` resolves `MODEL_ROOT = <repo>/models`. Add these files here
before deploying (they are tracked via Git LFS — see ../.gitattributes):

- Video_Fatigue.pth
- Landmark_Fatigue.pt
- Driver_Activity.pth
- Smoking_And_Drinking.pt
- SeatBelt_And_Phone.pt
- m4_normalization_stats_ws45.csv        (landmark fatigue normalization stats)
- face_landmarker.task                   (MediaPipe asset; auto-downloaded if missing)

Alternatively, uncomment the `hf_hub_download(...)` block near the top of app.py
to pull the weights from a separate Hugging Face model repo at startup instead
of committing them to the Space.
