"""
Driver Wellness AI — core inference module (ported from the Colab notebook).

This file is a faithful port of the notebook's model/adapter/fusion/orchestrator
code, with Colab-only pieces removed (Drive mount, input() prompt, %pip installs,
IPython inline display). Two glue functions are added at the bottom:

    build_manager()                       -> loads all 5 models, returns manager
    run_recorded_video(manager, path)     -> (annotated_mp4_path, summary_dict)

app.py imports and calls those two.
"""

from __future__ import annotations

import os
import gc
import math
import time
import json
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# tqdm: use the plain (headless-safe) variant, not tqdm.auto
from tqdm import tqdm

# IPython display is optional on Spaces; guard it so import never fails.
try:
    from IPython.display import clear_output as _ipy_clear_output
    from IPython.display import display as _ipy_display
    from IPython.display import Image as _IPyImage
except Exception:  # pragma: no cover
    def _ipy_clear_output(*a, **k): pass
    def _ipy_display(*a, **k): pass
    def _IPyImage(*a, **k): return None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

# ============================================================
# MODEL LOCATION — weights live in ./models next to this file
# ============================================================
MODEL_ROOT = Path(__file__).parent / "models"



# ==========================================================
# Driver Wellness AI — Configuration
# ==========================================================
from pathlib import Path

import torch

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
DRIVE_ROOT = MODEL_ROOT
# MODEL_ROOT is defined at the top of this file (repo-local ./models)

# ----------------------------------------------------------
# Input mode (set to "video" or "live" before Run All to skip prompt)
# ----------------------------------------------------------
INPUT_MODE = None  # None -> interactive prompt in Section 5

# ----------------------------------------------------------
# Device (auto-detected — never hardcoded)
# ----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Shared ImageNet normalization (Video Fatigue + Driver Activity backbones)
# ----------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------
# Video Fatigue — EfficientNet-B0 + BiLSTM (Section 7)
# ----------------------------------------------------------
VIDEO_FATIGUE_SEQUENCE_LENGTH = 16
VIDEO_FATIGUE_IMAGE_SIZE = (224, 224)
VIDEO_FATIGUE_HIDDEN_SIZE = 256
VIDEO_FATIGUE_NUM_LAYERS = 1
VIDEO_FATIGUE_BIDIRECTIONAL = True
VIDEO_FATIGUE_DROPOUT = 0.3
VIDEO_FATIGUE_CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
VIDEO_FATIGUE_RISK_MAP = {"Low Risk": 0.2, "Medium Risk": 0.6, "High Risk": 1.0}

# ----------------------------------------------------------
# Driver Activity — MobileNetV3-Large (Section 8)
# ----------------------------------------------------------
DRIVER_ACTIVITY_IMAGE_SIZE = (224, 224)
DRIVER_ACTIVITY_CLASS_NAMES = [
    "other_activities",
    "safe_driving",
    "talking_phone",
    "texting_phone",
    "turning",
]
DRIVER_ACTIVITY_RISK_MAP = {
    "safe_driving": 0.05,
    "turning": 0.20,
    "talking_phone": 0.60,
    "other_activities": 0.70,
    "texting_phone": 0.85,
}
# Frame-sampling stride used when running the (single-image) classifier over a video,
# matching the reused `VideoActivityProcessor(frame_skip=5)` default from the spec.
DRIVER_ACTIVITY_FRAME_SKIP = 5

# ----------------------------------------------------------
# Landmark Fatigue — MediaPipe + 2-layer LSTM (Section 9)
# ----------------------------------------------------------
LANDMARK_WINDOW_SIZE = 45
LANDMARK_FEATURE_NAMES = ["EAR", "MAR", "Pitch", "Yaw", "Roll"]
LANDMARK_INPUT_SIZE = 5
LANDMARK_HIDDEN_SIZE = 128
LANDMARK_NUM_LAYERS = 2
LANDMARK_NUM_CLASSES = 3
LANDMARK_ACTION_LABELS = ["Normal", "Talking", "Yawning"]
LANDMARK_FATIGUE_THRESHOLD = 0.05  # yawn_proportion >= this -> Drowsy (per spec; empirically validated in Milestone 5)
LANDMARK_NORMALIZATION_STATS_PATH = MODEL_ROOT / "m4_normalization_stats_ws45.csv"
MEDIAPIPE_FACE_LANDMARKER_PATH = MODEL_ROOT / "face_landmarker.task"
MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# ----------------------------------------------------------
# Smoking & Drinking — YOLOv8n (Section 10)
# ----------------------------------------------------------
SMOKING_IMGSZ = 640
SMOKING_CONF_THRESHOLD = 0.25
SMOKING_RISK_WEIGHTS = {"smoking": 0.8, "drinking": 0.5}
SMOKING_BOTH_DETECTED_RISK = 0.9

# ----------------------------------------------------------
# Seat Belt & Phone Usage Detection — YOLOv8n (Section 11)
# ----------------------------------------------------------
SEATBELT_PHONE_IMGSZ = 640
SEATBELT_PHONE_CONF_THRESHOLD = 0.25
SEATBELT_PHONE_CLASS_NAMES = {0: "Phone", 1: "Seatbelt"}

SEATBELT_CONF_PER_CLASS = {
    "Phone": 0.02,    # Ultra-low floor to pierce through shadows with temporal smoothing
    "Seatbelt": 0.35  # Strict floor to eliminate sunlight glare false positives
}

# Temporal Consensus Window Settings (in seconds and frame fractions)
SEATBELT_WINDOW_SEC = 0.50       # Temporal memory window size
SEATBELT_ON_FRAC = 0.40          # Fraction of frames required to confirm presence (lowered to catch brief phone flashes)
SEATBELT_OFF_FRAC = 0.20         # Hysteresis release threshold

# Updated Risk Map
SEATBELT_PHONE_RISK_MAP = {
    "Phone & Seatbelt": 0.85, # Distracted, but at least buckled
    "Phone Only": 1.0,        # Distracted AND unbuckled (Critical Risk!)
    "Seatbelt Only": 0.0,     # Safe and compliant
    "No Detection": 0.00      # Not distracted, but unbuckled (Moderate Risk)
}

# ----------------------------------------------------------
# Risk Fusion weights (Section 13)
# ----------------------------------------------------------
# TA feedback (M5): Video Fatigue standalone test accuracy was low (~33%, near random).
# Reduced from 0.35 -> 0.15 so a weak classifier cannot dominate the fused wellness score.
# Freed weight redistributed to landmark, driver activity, smoking, and seatbelt modules.
RISK_FUSION_WEIGHTS = {
    "video_fatigue": 0.20,
    "landmark_fatigue": 0.30,
    "driver_activity": 0.20,
    "smoking": 0.15,
    "seatbelt": 0.15,
}
assert abs(sum(RISK_FUSION_WEIGHTS.values()) - 1.0) < 1e-9, "Risk fusion weights must sum to 1.0"

# ----------------------------------------------------------
# Common Driver Risk Score Framework — Option A (PDF-pure)
# ----------------------------------------------------------
# Per risky prediction: R_i = severity_w_i × confidence. Safe predictions -> R_i = 0.
# overall_score = 100 × (1 - exp(-k × R_total))
FUSION_EXPONENTIAL_K = 0.05

# Severity weights aligned with Common_Driver_Risk_Score_Framework.pdf
RISK_EVENT_SEVERITY: Dict[str, Dict[str, float]] = {
    "video_fatigue": {
        "Medium Risk": 6.0,
        "High Risk": 10.0,
    },
    "landmark_fatigue": {
        "Drowsy": 7.0,
        "Yawning": 7.0,
        "Talking": 4.0,
    },
    "driver_activity": {
        "turning": 4.0,
        "talking_phone": 8.0,
        "texting_phone": 9.0,
        "other_activities": 6.0,
    },
    "smoking": {
        "smoking": 5.0,
        "drinking": 10.0,
        "smoking+drinking": 10.0,
    },
    "seatbelt": {
        "Phone Only": 9.0,
        "Phone & Seatbelt": 9.0,
        "No Detection": 8.0,
    },
}

# Predictions explicitly treated as safe (R_i = 0)
RISK_EVENT_SAFE_PREDICTIONS: Dict[str, set] = {
    "video_fatigue": {"Low Risk"},
    "landmark_fatigue": {"Alert", "Normal"},
    "driver_activity": {"safe_driving"},
    "smoking": {"none"},
    "seatbelt": {"Seatbelt Only"},
}

FUSION_MODULE_NAME_TO_KEY: Dict[str, str] = {
    "Video Fatigue Detection": "video_fatigue",
    "Landmark Fatigue Detection": "landmark_fatigue",
    "Driver Activity Recognition": "driver_activity",
    "Smoking & Drinking Detection": "smoking",
    "Seat Belt & Phone Usage Detection": "seatbelt",
}

# Optional temporal smoothing on fused score (PDF Step 5)
FUSION_SCORE_SMOOTHING_ENABLED = True
FUSION_SCORE_SMOOTHING_WINDOW_SEC = 3.0

print("=" * 80)
print("Driver Wellness AI — Configuration Loaded")
print("=" * 80)
print(f"Device      : {DEVICE}")
print(f"MODEL_ROOT  : {MODEL_ROOT}")
print(f"Risk Fusion Weights : {RISK_FUSION_WEIGHTS}")
print(f"Fusion mode : Option A exponential (k={FUSION_EXPONENTIAL_K})")

# ----------------------------------------------------------
# Smoking & Drinking — TEMPORAL-CONSISTENCY LAYER (Section 10)
# ----------------------------------------------------------
# The YOLOv8n detector fires per-frame, so a genuine action and a 2-3 frame
# flicker look identical to it. A sliding-window majority vote + hysteresis is
# layered on top so a class is only COUNTED/DRAWN once it is seen consistently
# across a short time window, and only released once it has been absent for a
# while. Tuned to BALANCE false positives (flicker suppressed) against false
# negatives (a sustained real action is confirmed and held via hysteresis).
SMOKING_CONF_PER_CLASS = {          # per-class raw-confidence floor (before voting)
    "smoking": 0.25,
    "drinking": 0.40,               # drinking flickers more, so a slightly higher floor
}
SMOKING_WINDOW_SEC = 0.50           # sliding-window length, in SECONDS (fps-independent)
SMOKING_ON_FRAC = 0.50              # >= this fraction of the window are hits -> CONFIRM (ON)
SMOKING_OFF_FRAC = 0.30             # <= this fraction -> release (OFF). Gap = hysteresis.
# Stricter (WINDOW=0.9, ON=0.6) -> fewer false positives, slower to react (more FN).
# Looser  (WINDOW=0.3, ON=0.4) -> reacts faster, but more false positives.

# ----------------------------------------------------------
# Streaming Configuration (TA sliding-window simulation)
# ----------------------------------------------------------
STREAMING_MODE = True                     # preferred for integrated / long-video demo
VIDEO_FATIGUE_WINDOW = 16
VIDEO_FATIGUE_STRIDE = 4                  # re-infer every N new frames once buffer is full
LANDMARK_STRIDE = 5
DRIVER_ACTIVITY_HISTORY = 10              # rolling probability history (online smoothing)
SEATBELT_PHONE_FRAME_STRIDE = 5
SMOKING_FRAME_STRIDE = 3                  # was 1 (every frame); loosened for live-webcam throughput.
                                           # TemporalVoter's window/on/off hit counts are unaffected
                                           # (still computed from fps*window_sec), but since YOLO now
                                           # only runs on every 3rd frame, the window spans ~3x more
                                           # wall-clock time -> confirm/release reacts ~3x slower.
SEGMENT_SUMMARY_SEC = 15                  # reporting interval — does NOT reset buffers
FUSION_UPDATE_INTERVAL_SEC = 1.0
STORE_FULL_TIMELINE = False
TIMELINE_SAMPLE_EVERY_N_FRAMES = 30
MAX_TIMELINE_ENTRIES = 2000
ENABLE_LIVE_DISPLAY = False  # no inline display on Spaces
DISPLAY_MAX_FPS = 2
ENABLE_MEMORY_DIAGNOSTICS = False
SAVE_SEGMENT_SUMMARIES = False
RECOVERY_OUTPUT_DIR = Path(tempfile.gettempdir()) / "streaming_recovery"

# ---------- CONFIG (edit these) ----------
LIVE_MAX_DURATION_MIN = 1      # auto-stop after N minutes (default 10)
LIVE_WEBCAM_FPS = 15.0          # timestamp assumption for fusion timeline
LIVE_FRAME_DELAY_SEC = 0.05     # ~20 fps max in Colab JS path
LIVE_MAX_FRAMES = None          # optional hard cap; None = time limit only
# e.g. LIVE_MAX_FRAMES = 900    # ~30s safety cap if you want both



# ==========================================================
# Driver Wellness AI — Core SDK
# ==========================================================
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("DriverWellnessAI")


# ----------------------------------------------------------
# Exceptions (shared across every adapter)
# ----------------------------------------------------------
class AdapterError(Exception):
    """Base class for every Driver Wellness AI adapter error."""


class CheckpointNotFoundError(AdapterError, FileNotFoundError):
    """Raised when a model checkpoint file cannot be located on disk."""


class VideoNotFoundError(AdapterError, FileNotFoundError):
    """Raised when the input video file cannot be located on disk."""


class VideoDecodeError(AdapterError, RuntimeError):
    """Raised when OpenCV cannot open or decode the given video file."""


class InsufficientBufferError(AdapterError, RuntimeError):
    """Raised when fewer frames are available than a module's minimum window size."""


class FaceNotDetectedError(AdapterError, RuntimeError):
    """Raised when the face landmark detector fails to find a face where required."""


class InvalidInputError(AdapterError, ValueError):
    """Raised when input data is not a valid ndarray, or has the wrong shape/dtype."""


class ModelNotLoadedError(AdapterError, RuntimeError):
    """Raised when predict() is invoked but the underlying model failed to load."""


class GPUOutOfMemoryError(AdapterError, RuntimeError):
    """Raised when a CUDA out-of-memory error occurs during inference."""


# ----------------------------------------------------------
# Native-type conversion (keeps PredictionResult free of numpy/torch types)
# ----------------------------------------------------------
def to_native_types(value: Any) -> Any:
    """Recursively converts numpy/torch types into native Python objects."""
    if isinstance(value, dict):
        return {key: to_native_types(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native_types(item) for item in value]
    if isinstance(value, torch.Tensor):
        return to_native_types(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_native_types(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ----------------------------------------------------------
# PredictionResult
# ----------------------------------------------------------
@dataclass
class PredictionResult:
    """
    Standardized output produced by every Driver Wellness AI adapter.

    This dataclass is the ONLY thing `DriverWellnessModuleManager` and
    `DriverWellnessRiskFusion` know about — neither understands anything
    about any individual model's architecture, preprocessing, or output
    format beyond this shape.
    """

    module: str
    prediction: str
    confidence: float
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to a plain dict following the mandatory output contract."""
        return {
            "module": self.metadata.get("module_key", self.module),
            "prediction": str(self.prediction),
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
            "status": "ERROR" if self.is_error else "OK",
        }

    @property
    def is_error(self) -> bool:
        """True if this result represents a failed module run rather than a real prediction."""
        return self.error is not None


# ----------------------------------------------------------
# InferenceContext
# ----------------------------------------------------------
@dataclass
class InferenceContext:
    """
    Carries everything an adapter needs to run one inference pass, decoupling
    every adapter from *how* the driving video reached the pipeline (upload
    cell, Drive path, live buffer, etc).
    """

    video_path: str
    device: torch.device = field(default_factory=lambda: DEVICE)
    session_id: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not Path(self.video_path).is_file():
            raise VideoNotFoundError(f"Video not found: {self.video_path}")


# ----------------------------------------------------------
# BaseModelAdapter
# ----------------------------------------------------------
class BaseModelAdapter(ABC):
    """
    Abstract base class for every Driver Wellness AI inference adapter.

    Every adapter wraps a single, independently trained model and exposes a
    uniform lifecycle (lazy `load_model`, `preprocess`, `predict`, `warmup`,
    `unload`) plus a uniform `PredictionResult` output, so that
    `DriverWellnessModuleManager` can treat every module identically without
    any adapter-specific branching.
    """

    #: Registry key used throughout the pipeline (e.g. "video_fatigue").
    MODULE_KEY: str = "base_adapter"
    #: Human-readable name stamped into every PredictionResult.module.
    MODULE_NAME: str = "Base Adapter"

    def __init__(
        self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None
    ) -> None:
        """Initializes the adapter WITHOUT loading the model (lazy loading only)."""
        self.checkpoint_path: Path = Path(checkpoint_path)
        self.device: torch.device = device if device is not None else DEVICE
        self.model: Optional[Any] = None
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_model(self) -> None:
        """Loads the underlying model and weights into `self.model`."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, context: InferenceContext) -> Any:
        """Transforms the raw input referenced by `context` into a model-ready representation."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: InferenceContext) -> PredictionResult:
        """Runs inference end-to-end and returns a `PredictionResult`."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        """Loads the model (if needed) and runs one dummy inference pass."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Releases the model and any device memory held by this adapter."""
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        """Lazily loads the model on first use. Never invoked from `__init__`."""
        if self.model is None:
            self._logger.info("%s model not yet loaded; loading now.", self.MODULE_NAME)
            self.load_model()

    def build_result(
        self,
        prediction: str,
        confidence: float,
        risk_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Builds a `PredictionResult`, guaranteeing every value is a native Python type."""
        return PredictionResult(
            module=self.MODULE_NAME,
            prediction=str(prediction),
            confidence=float(confidence),
            risk_score=float(risk_score),
            metadata=to_native_types(metadata or {}),
        )

    def error_result(self, exc: Exception) -> PredictionResult:
        """Builds a `PredictionResult` representing a failed module run (never re-raises)."""
        self._logger.error("%s failed: %s", self.MODULE_NAME, exc)
        return PredictionResult(
            module=self.MODULE_NAME,
            prediction="Error",
            confidence=0.0,
            risk_score=0.0,
            metadata={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            error=str(exc),
        )


# ----------------------------------------------------------
# AdapterRegistry
# ----------------------------------------------------------
class AdapterRegistry:
    """
    Lightweight registry mapping module keys to adapter *classes*.

    Decouples "which adapter class implements module X" from the manager
    that runs adapters, so new adapters can be added without ever touching
    `DriverWellnessModuleManager`.
    """

    def __init__(self) -> None:
        self._adapter_classes: Dict[str, Type[BaseModelAdapter]] = {}

    def register(self, module_key: str, adapter_cls: Type[BaseModelAdapter]) -> None:
        """Registers an adapter CLASS under `module_key`."""
        if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, BaseModelAdapter)):
            raise TypeError(f"{adapter_cls!r} must be a BaseModelAdapter subclass")
        self._adapter_classes[module_key] = adapter_cls

    def get(self, module_key: str) -> Type[BaseModelAdapter]:
        """Returns the adapter class registered under `module_key`."""
        if module_key not in self._adapter_classes:
            raise KeyError(f"No adapter class registered for module key: {module_key}")
        return self._adapter_classes[module_key]

    def keys(self) -> List[str]:
        """Returns every registered module key."""
        return list(self._adapter_classes.keys())


# ----------------------------------------------------------
# DriverWellnessModuleManager
# ----------------------------------------------------------
class DriverWellnessModuleManager:
    """
    Central orchestrator for the Driver Wellness AI pipeline.

    Responsibilities (and ONLY these — no AI/model-specific logic lives
    here):
        - Register adapter instances
        - Load models for every registered adapter
        - Execute every adapter against a shared `InferenceContext`
        - Collect the resulting `PredictionResult` objects
        - Return the aggregated results

    This class never inspects an adapter's internals and never branches on
    `module_key` — every adapter is driven purely through the
    `BaseModelAdapter` interface.
    """

    def __init__(self, registry: Optional[AdapterRegistry] = None) -> None:
        self.registry: AdapterRegistry = registry or AdapterRegistry()
        self._instances: Dict[str, BaseModelAdapter] = {}

    def register_adapter(self, module_key: str, adapter: BaseModelAdapter) -> None:
        """Registers a ready-to-use adapter INSTANCE under `module_key`."""
        if not isinstance(adapter, BaseModelAdapter):
            raise TypeError("adapter must be a BaseModelAdapter instance")
        self._instances[module_key] = adapter
        self.registry.register(module_key, type(adapter))
        logger.info("Registered adapter '%s' -> %s", module_key, adapter.MODULE_NAME)

    def get_adapter(self, module_key: str) -> BaseModelAdapter:
        """Returns the registered adapter instance for `module_key`."""
        return self._instances[module_key]

    def registered_modules(self) -> List[str]:
        """Returns every registered module key, in registration order."""
        return list(self._instances.keys())

    def load_all(self) -> None:
        """Triggers model loading (via `warmup()`) for every registered adapter."""
        for module_key, adapter in self._instances.items():
            try:
                logger.info("Loading model for '%s'...", module_key)
                adapter.warmup()
            except Exception as exc:  # noqa: BLE001 - a load failure must not abort the pipeline
                logger.error("Failed to load/warm up '%s': %s", module_key, exc)

    def run_all(self, context: InferenceContext) -> List[PredictionResult]:
        """
        Executes every registered adapter against `context`.

        A single adapter's failure never aborts the pipeline: it is captured
        as an error `PredictionResult` (via `adapter.error_result`) instead
        of propagating.

        Args:
            context: The shared `InferenceContext` for this inference run.

        Returns:
            One `PredictionResult` per registered adapter.
        """
        results: List[PredictionResult] = []
        for module_key, adapter in self._instances.items():
            logger.info("Running adapter '%s'...", module_key)
            try:
                result = adapter.predict(context)
            except Exception as exc:  # noqa: BLE001 - isolate failures per-module
                result = adapter.error_result(exc)
            result.weight = RISK_FUSION_WEIGHTS.get(module_key, 0.0)
            result.metadata = dict(result.metadata or {})
            result.metadata["module_key"] = module_key
            results.append(result)
        return results

    def unload_all(self) -> None:
        """Releases every registered adapter's model and device memory."""
        for module_key, adapter in self._instances.items():
            try:
                adapter.unload()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to unload '%s': %s", module_key, exc)


print("=" * 80)
print("Driver Wellness AI — Core SDK Loaded")
print("=" * 80)
print("Defined: PredictionResult, InferenceContext, BaseModelAdapter, AdapterRegistry, "
      "DriverWellnessModuleManager")



# ==========================================================
# Driver Wellness AI — Shared Video Preprocessing Helpers
# (reused, unmodified, by Video Fatigue AND Driver Activity adapters)
# ==========================================================
import gc
from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def read_video_frames(video_path: Union[str, Path], frame_skip: int = 1) -> List[np.ndarray]:
    """
    Reads frames from a video file using OpenCV, honoring `frame_skip`.

    Raises:
        VideoDecodeError: If OpenCV cannot open the video container.
        InsufficientBufferError: If the video opens but yields zero frames.
    """
    if frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError(f"Cannot decode video (unsupported/corrupted file): {video_path}")

    frames: List[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_skip == 0:
                frames.append(frame)
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise InsufficientBufferError(f"No frames could be extracted from video: {video_path}")

    return frames


def sample_frame_sequence(frames: Sequence[np.ndarray], sequence_length: int) -> List[np.ndarray]:
    """
    Produces exactly `sequence_length` frames: uniform temporal sampling if
    more frames are available, or padding with the final frame if fewer.
    """
    total_frames = len(frames)
    if total_frames == sequence_length:
        return list(frames)

    if total_frames > sequence_length:
        indices = np.round(np.linspace(0, total_frames - 1, num=sequence_length)).astype(int)
        return [frames[index] for index in indices]

    logger.warning(
        "Video yielded only %d frames but %d are required; padding with the final frame.",
        total_frames,
        sequence_length,
    )
    padded_frames = list(frames)
    while len(padded_frames) < sequence_length:
        padded_frames.append(frames[-1])
    return padded_frames


def frames_to_tensor(
    frames: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """Converts a sequence of BGR frames into a normalized `(T, C, H, W)` tensor."""
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    tensor_frames = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frames.append(transform(Image.fromarray(rgb_frame)))
    return torch.stack(tensor_frames, dim=0)


print("Shared video preprocessing helpers loaded: "
      "read_video_frames, sample_frame_sequence, frames_to_tensor")



# ==========================================================
# Driver Wellness AI — Video Fatigue Adapter
# EfficientNet-B0 -> BiLSTM -> Linear (Low / Medium / High Risk)
# ==========================================================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetBiLSTM(nn.Module):
    """
    EfficientNet-B0 CNN backbone + BiLSTM temporal head + linear classifier.

    Architecture reconstructed EXACTLY from the real `Video_Fatigue.pth`
    checkpoint's `model_state_dict` (`feature_extractor.*`, `lstm.*`,
    `classifier.*` — 368 tensors, verified via `load_state_dict(strict=True)`
    with zero missing/unexpected keys). The checkpoint stores
    `feature_extractor` as EfficientNet-B0's bare `.features` submodule (the
    convolutional stack only — no built-in pooling or classifier head), NOT
    the full `EfficientNet` model. Per-frame visual features are extracted
    independently by this shared backbone, globally average-pooled, and the
    resulting per-frame feature sequence is fed through a bidirectional LSTM
    to model temporal fatigue dynamics; the final timestep's hidden state is
    projected to class logits via a linear classification head.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None)
        self.feature_dim: int = efficientnet.classifier[1].in_features
        self.feature_extractor = efficientnet.features

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (batch, seq_len, 3, H, W). Returns logits (batch, num_classes)."""
        batch_size, seq_len, channels, height, width = x.shape
        frames = x.view(batch_size * seq_len, channels, height, width)
        features = self.feature_extractor(frames)
        features = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        features = features.view(batch_size, seq_len, -1)
        sequence_output, _ = self.lstm(features)
        last_step = self.dropout(sequence_output[:, -1, :])
        return self.classifier(last_step)


class VideoFatigueAdapter(BaseModelAdapter):
    """
    Inference adapter for the Video Fatigue model (`Video_Fatigue.pth`).

    Wraps the existing EfficientNet-B0 + BiLSTM + Linear implementation:
    checkpoint loading, 16-frame sequence sampling, ImageNet normalization,
    softmax confidence, and the Low/Medium/High risk mapping are preserved
    exactly as specified — nothing about the model itself is changed here.
    """

    MODULE_KEY: str = "video_fatigue"
    MODULE_NAME: str = "Video Fatigue Detection"

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.class_names: List[str] = VIDEO_FATIGUE_CLASS_NAMES
        self.risk_map: Dict[str, float] = VIDEO_FATIGUE_RISK_MAP
        self.sequence_length: int = VIDEO_FATIGUE_SEQUENCE_LENGTH
        self.image_size: Tuple[int, int] = VIDEO_FATIGUE_IMAGE_SIZE

    def load_model(self) -> None:
        """Loads `EfficientNetBiLSTM` and its trained weights onto `self.device`."""
        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"[{self.MODULE_NAME}] Checkpoint not found: {self.checkpoint_path}")

        self._logger.info("[%s] Loading checkpoint from %s", self.MODULE_NAME, self.checkpoint_path)
        model = EfficientNetBiLSTM(
            num_classes=len(self.class_names),
            hidden_size=VIDEO_FATIGUE_HIDDEN_SIZE,
            num_layers=VIDEO_FATIGUE_NUM_LAYERS,
            bidirectional=VIDEO_FATIGUE_BIDIRECTIONAL,
            dropout=VIDEO_FATIGUE_DROPOUT,
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        self._logger.info("[%s] Model loaded on %s", self.MODULE_NAME, self.device)

    def preprocess(self, context: InferenceContext) -> torch.Tensor:
        """Converts the video referenced by `context` into a `(1, T, 3, H, W)` tensor."""
        raw_frames = read_video_frames(context.video_path, frame_skip=1)
        sampled_frames = sample_frame_sequence(raw_frames, self.sequence_length)
        sequence_tensor = frames_to_tensor(sampled_frames, self.image_size, IMAGENET_MEAN, IMAGENET_STD)
        return sequence_tensor.unsqueeze(0)

    def predict(self, context: InferenceContext) -> PredictionResult:
        """Runs end-to-end inference and returns a `PredictionResult`."""
        self._ensure_loaded()
        input_tensor = self.preprocess(context).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        confidence_value, predicted_index = torch.max(probabilities, dim=0)
        predicted_class = self.class_names[int(predicted_index.item())]
        confidence = float(confidence_value.item())
        risk_score = float(self.risk_map[predicted_class])

        class_probabilities = {
            name: float(prob) for name, prob in zip(self.class_names, probabilities.detach().cpu().tolist())
        }

        return self.build_result(
            prediction=predicted_class,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "class_probabilities": class_probabilities,
                "sequence_length": self.sequence_length,
                "image_size": list(self.image_size),
                "checkpoint": str(self.checkpoint_path),
            },
        )


    # --- Streaming API (bounded rolling buffer — legacy batch predict() unchanged) ---
    def reset_streaming_state(self) -> None:
        from collections import deque
        self._stream_buffer = deque(maxlen=self.sequence_length)
        self._stream_since_infer = 0
        self._stream_latest: Optional[PredictionResult] = None
        self._stream_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ])

    def _preprocess_stream_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._stream_transform(Image.fromarray(rgb_frame))

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float) -> Dict[str, Any]:
        """Append one preprocessed frame; infer on a rolling window when ready."""
        self._ensure_loaded()
        if not hasattr(self, "_stream_buffer"):
            self.reset_streaming_state()
        self._stream_buffer.append(self._preprocess_stream_frame(frame_bgr))
        self._stream_since_infer += 1
        buffer_len = len(self._stream_buffer)
        if buffer_len < self.sequence_length:
            return {"status": "WARMING_UP", "result": None, "buffer_len": buffer_len}
        if self._stream_since_infer % VIDEO_FATIGUE_STRIDE != 0 and self._stream_latest is not None:
            return {"status": "READY", "result": self._stream_latest, "buffer_len": buffer_len}
        sequence_tensor = torch.stack(list(self._stream_buffer), dim=0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(sequence_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
        confidence_value, predicted_index = torch.max(probabilities, dim=0)
        predicted_class = self.class_names[int(predicted_index.item())]
        self._stream_latest = self.build_result(
            prediction=predicted_class,
            confidence=float(confidence_value.item()),
            risk_score=float(self.risk_map[predicted_class]),
            metadata={
                "class_probabilities": {
                    name: float(prob)
                    for name, prob in zip(self.class_names, probabilities.detach().cpu().tolist())
                },
                "sequence_length": self.sequence_length,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "buffer_len": buffer_len,
                "mode": "streaming",
            },
        )
        return {"status": "READY", "result": self._stream_latest, "buffer_len": buffer_len}

    def warmup(self) -> None:
        """Loads the model (if needed) and runs a single dummy forward pass."""
        self._ensure_loaded()
        dummy_input = torch.zeros(
            (1, self.sequence_length, 3, self.image_size[0], self.image_size[1]), device=self.device
        )
        with torch.no_grad():
            self.model(dummy_input)
        self._logger.info("[%s] Warmup completed", self.MODULE_NAME)

    def unload(self) -> None:
        """Releases the model and frees device memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("VideoFatigueAdapter ready.")



# ==========================================================
# Driver Wellness AI — Driver Activity Adapter
# MobileNetV3-Large -> Linear(1280, 5)
# ==========================================================
class DriverActivityAdapter(BaseModelAdapter):
    """
    Inference adapter for the Driver Activity model (`Driver_Activity.pth`).

    Wraps the existing MobileNetV3-Large classifier exactly as specified:
    `models.mobilenet_v3_large(weights=None)` with `classifier[3]` replaced
    by `Linear(1280, 5)`, ImageNet preprocessing, and `softmax` + `argmax`
    confidence. Video frames are sampled and averaged at the integration
    layer only (see module docstring above) — the per-frame model logic is
    untouched.
    """

    MODULE_KEY: str = "driver_activity"
    MODULE_NAME: str = "Driver Activity Recognition"

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.class_names: List[str] = list(DRIVER_ACTIVITY_CLASS_NAMES)
        self.risk_map: Dict[str, float] = DRIVER_ACTIVITY_RISK_MAP
        self.image_size: Tuple[int, int] = DRIVER_ACTIVITY_IMAGE_SIZE
        self.frame_skip: int = DRIVER_ACTIVITY_FRAME_SKIP
        self._transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            ]
        )

    def load_model(self) -> None:
        """Loads MobileNetV3-Large and its trained weights, per the checkpoint spec."""
        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"[{self.MODULE_NAME}] Checkpoint not found: {self.checkpoint_path}")

        self._logger.info("[%s] Loading checkpoint from %s", self.MODULE_NAME, self.checkpoint_path)
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(in_features=1280, out_features=len(self.class_names))

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "class_names" in checkpoint and checkpoint["class_names"]:
                self.class_names = list(checkpoint["class_names"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self.model = model
        self._logger.info("[%s] Model loaded on %s", self.MODULE_NAME, self.device)

    def preprocess(self, context: InferenceContext) -> List[np.ndarray]:
        """Samples frames from the video at `frame_skip` intervals (RGB numpy arrays)."""
        raw_frames = read_video_frames(context.video_path, frame_skip=self.frame_skip)
        return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in raw_frames]

    def _predict_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Runs the classifier on a single RGB frame; returns the softmax probability vector."""
        image = Image.fromarray(frame_rgb)
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        return probabilities.detach().cpu().numpy()

    def predict(self, context: InferenceContext) -> PredictionResult:
        """Runs per-frame inference across the video and aggregates into one video-level result."""
        self._ensure_loaded()
        sampled_frames = self.preprocess(context)

        timeline: List[Dict[str, Any]] = []
        probability_sum = np.zeros(len(self.class_names), dtype=np.float64)

        for frame_index, frame_rgb in enumerate(sampled_frames):
            probabilities = self._predict_frame(frame_rgb)
            probability_sum += probabilities
            frame_predicted_index = int(np.argmax(probabilities))
            timeline.append(
                {
                    "frame_index": frame_index * self.frame_skip,
                    "predicted_class": self.class_names[frame_predicted_index],
                    "confidence": float(probabilities[frame_predicted_index]),
                }
            )

        mean_probabilities = probability_sum / len(sampled_frames)
        predicted_index = int(np.argmax(mean_probabilities))
        predicted_class = self.class_names[predicted_index]
        confidence = float(mean_probabilities[predicted_index])  # 0.0-1.0

        # Exact risk formula from the Module Manager Contract (Section 10),
        # with confidence already normalized to 0-1 (see module docstring).
        base_risk = self.risk_map.get(predicted_class, 0.50)
        risk_score = float(base_risk * confidence)

        probabilities_dict = {
            name: float(prob) for name, prob in zip(self.class_names, mean_probabilities.tolist())
        }

        return self.build_result(
            prediction=predicted_class,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "predicted_index": predicted_index,
                "probabilities": probabilities_dict,
                "confidence_percent": confidence * 100.0,
                "frames_sampled": len(sampled_frames),
                "frame_skip": self.frame_skip,
                "timeline": timeline,
                "checkpoint": str(self.checkpoint_path),
            },
        )


    # --- Streaming API (legacy batch predict() unchanged) ---
    def reset_streaming_state(self) -> None:
        from collections import deque
        self._stream_prob_history = deque(maxlen=DRIVER_ACTIVITY_HISTORY)
        self._stream_latest: Optional[PredictionResult] = None

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float) -> Dict[str, Any]:
        self._ensure_loaded()
        if not hasattr(self, "_stream_prob_history"):
            self.reset_streaming_state()
        if frame_index % self.frame_skip != 0 and self._stream_latest is not None:
            return {"status": "READY", "result": self._stream_latest}
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        probabilities = self._predict_frame(frame_rgb)
        self._stream_prob_history.append(probabilities)
        mean_probabilities = np.mean(np.stack(list(self._stream_prob_history), axis=0), axis=0)
        predicted_index = int(np.argmax(mean_probabilities))
        predicted_class = self.class_names[predicted_index]
        confidence = float(mean_probabilities[predicted_index])
        base_risk = self.risk_map.get(predicted_class, 0.50)
        self._stream_latest = self.build_result(
            prediction=predicted_class,
            confidence=confidence,
            risk_score=float(base_risk * confidence),
            metadata={
                "probabilities": {
                    name: float(prob) for name, prob in zip(self.class_names, mean_probabilities.tolist())
                },
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "history_len": len(self._stream_prob_history),
                "mode": "streaming",
            },
        )
        return {"status": "READY", "result": self._stream_latest}

    def warmup(self) -> None:
        """Loads the model (if needed) and runs a single dummy forward pass."""
        self._ensure_loaded()
        dummy_input = torch.zeros((1, 3, self.image_size[0], self.image_size[1]), device=self.device)
        with torch.no_grad():
            self.model(dummy_input)
        self._logger.info("[%s] Warmup completed", self.MODULE_NAME)

    def unload(self) -> None:
        """Releases the model and frees device memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("DriverActivityAdapter ready.")



# ==========================================================
# Driver Wellness AI — Landmark Adapter: MediaPipe feature extraction
# ==========================================================
import math
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Standard MediaPipe Face Mesh (468/478-point) landmark indices for EAR/MAR/head-pose.
# These are the well-known public indices used throughout the fatigue-detection
# literature for these exact named features (not model-specific / proprietary).
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_MAR_IDX = {"corner_left": 61, "corner_right": 291, "lip_upper": 39, "lip_lower": 0}


def _ensure_face_landmarker_asset() -> Path:
    """Downloads the MediaPipe Face Landmarker `.task` asset if not already present."""
    if not MEDIAPIPE_FACE_LANDMARKER_PATH.is_file():
        MEDIAPIPE_FACE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe Face Landmarker task file...")
        urllib.request.urlretrieve(MEDIAPIPE_FACE_LANDMARKER_URL, str(MEDIAPIPE_FACE_LANDMARKER_PATH))
    return MEDIAPIPE_FACE_LANDMARKER_PATH


def create_face_landmarker() -> FaceLandmarker:
    """Creates a single-face MediaPipe Face Landmarker in IMAGE running mode.

    The CPU delegate is forced explicitly: this is a lightweight per-frame
    landmark model that does not need GPU acceleration, and forcing CPU
    avoids platform-specific GPU/EGL/Metal context-creation requirements
    that can otherwise fail in headless notebook environments.
    """
    asset_path = _ensure_face_landmarker_asset()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(asset_path), delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def _euclidean(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


def _landmark_xy(landmark, image_w: int, image_h: int) -> Tuple[float, float]:
    return landmark.x * image_w, landmark.y * image_h


def compute_ear(landmarks, image_w: int, image_h: int) -> float:
    """Eye Aspect Ratio, averaged across both eyes (Soukupová & Čech, 2016)."""

    def _ear_for_eye(indices: List[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [_landmark_xy(landmarks[i], image_w, image_h) for i in indices]
        vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
        horizontal = 2.0 * _euclidean(p1, p4)
        return vertical / horizontal if horizontal > 1e-6 else 0.0

    return (_ear_for_eye(LEFT_EYE_EAR_IDX) + _ear_for_eye(RIGHT_EYE_EAR_IDX)) / 2.0


def compute_mar(landmarks, image_w: int, image_h: int) -> float:
    """Mouth Aspect Ratio: vertical lip opening over horizontal mouth width."""
    upper = _landmark_xy(landmarks[MOUTH_MAR_IDX["lip_upper"]], image_w, image_h)
    lower = _landmark_xy(landmarks[MOUTH_MAR_IDX["lip_lower"]], image_w, image_h)
    left = _landmark_xy(landmarks[MOUTH_MAR_IDX["corner_left"]], image_w, image_h)
    right = _landmark_xy(landmarks[MOUTH_MAR_IDX["corner_right"]], image_w, image_h)
    vertical = _euclidean(upper, lower)
    horizontal = _euclidean(left, right)
    return vertical / horizontal if horizontal > 1e-6 else 0.0


def compute_head_pose(matrix) -> Tuple[float, float, float]:
    """Head Pitch/Yaw/Roll (degrees), decomposed from MediaPipe's own
    facial transformation matrix (matches the Milestone 4 training pipeline)."""
    rmat = np.array(matrix)[:3, :3]
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
    yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    return pitch, yaw, roll


def extract_landmark_features(face_landmarker: FaceLandmarker, frame_bgr: np.ndarray) -> Optional[List[float]]:
    """Returns `[EAR, MAR, Pitch, Yaw, Roll]` for one frame, or `None` if no face was detected."""
    image_h, image_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks or not result.facial_transformation_matrixes:
        return None

    landmarks = result.face_landmarks[0]
    ear = compute_ear(landmarks, image_w, image_h)
    mar = compute_mar(landmarks, image_w, image_h)
    pitch, yaw, roll = compute_head_pose(result.facial_transformation_matrixes[0])
    return [ear, mar, pitch, yaw, roll]


def load_normalization_stats(path: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads per-feature normalization mean/std from `m4_normalization_stats_ws45.csv`.

    Accepts either a "long" schema (columns `feature`, `mean`, `std`) or a "wide"
    schema (one row each for mean/std, one column per feature). Falls back to
    identity normalization (mean=0, std=1) with a warning if the file is missing
    or unrecognized — replace with the real training-set statistics for accurate
    inference.
    """
    if not path.is_file():
        logger.warning(
            "Normalization stats file not found at %s; using identity normalization.", path
        )
        return np.zeros(len(feature_names)), np.ones(len(feature_names))

    try:
        stats_df = pd.read_csv(path)
        if {"feature", "mean", "std"}.issubset(stats_df.columns):
            stats_df = stats_df.set_index("feature").loc[feature_names]
            return stats_df["mean"].to_numpy(dtype=np.float64), stats_df["std"].to_numpy(dtype=np.float64)
        if set(feature_names).issubset(stats_df.columns) and len(stats_df) >= 2:
            mean_row = stats_df.iloc[0][feature_names].to_numpy(dtype=np.float64)
            std_row = stats_df.iloc[1][feature_names].to_numpy(dtype=np.float64)
            return mean_row, std_row
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse normalization stats at %s (%s); using identity.", path, exc)

    logger.warning("Unrecognized normalization stats schema at %s; using identity normalization.", path)
    return np.zeros(len(feature_names)), np.ones(len(feature_names))


print("Landmark feature-extraction utilities loaded.")



# ==========================================================
# Driver Wellness AI — Landmark Adapter
# 2-layer LSTM (input=5, hidden=128) over 45-frame EAR/MAR/pose windows
# ==========================================================
# Yawning-class F1 reliability gate: per the spec, currently measured at 0.595
# (> the 0.50 threshold), so fatigue-state derivation is currently trusted.
LANDMARK_RELIABILITY_GATE_OPEN = True


class LSTMClassifier(nn.Module):
    """2-layer LSTM classifying 45-frame landmark-feature windows into an action label."""

    def __init__(self, input_size: int = 5, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (batch, window_size, input_size). Returns logits (batch, num_classes)."""
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


class LandmarkFatigueAdapter(BaseModelAdapter):
    """
    Inference adapter for the Landmark Fatigue model (`Landmark_Fatigue.pt`).

    Wraps the existing pipeline: MediaPipe feature extraction -> normalization
    -> 45-frame windowing -> 2-layer LSTM -> softmax -> rule-based fatigue
    mapping, exactly per the Technical Integration Specification / Module
    Manager Contract.
    """

    MODULE_KEY: str = "landmark_fatigue"
    MODULE_NAME: str = "Landmark Fatigue Detection"

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.window_size: int = LANDMARK_WINDOW_SIZE
        self.action_labels: List[str] = LANDMARK_ACTION_LABELS
        self.feature_names: List[str] = LANDMARK_FEATURE_NAMES
        self.face_landmarker: Optional[FaceLandmarker] = None
        self.normalization_mean, self.normalization_std = load_normalization_stats(
            LANDMARK_NORMALIZATION_STATS_PATH, self.feature_names
        )
        # Per-instance (NOT module-level global) history -- fixes the thread-safety
        # issue flagged in the Module Manager Contract, Section 8.
        self._recent_action_history: List[str] = []

    def load_model(self) -> None:
        """Loads the 2-layer LSTM weights and initializes the MediaPipe Face Landmarker."""
        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"[{self.MODULE_NAME}] Checkpoint not found: {self.checkpoint_path}")

        self._logger.info("[%s] Loading checkpoint from %s", self.MODULE_NAME, self.checkpoint_path)
        model = LSTMClassifier(
            input_size=LANDMARK_INPUT_SIZE,
            hidden_size=LANDMARK_HIDDEN_SIZE,
            num_layers=LANDMARK_NUM_LAYERS,
            num_classes=LANDMARK_NUM_CLASSES,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        self.face_landmarker = create_face_landmarker()
        self._logger.info("[%s] Model + Face Landmarker loaded on %s", self.MODULE_NAME, self.device)

    def preprocess(self, context: InferenceContext) -> Tuple[List[np.ndarray], List[bool], List[bool]]:
        """
        Extracts per-frame EAR/MAR/Pitch/Yaw/Roll, normalizes, and splits into
        non-overlapping 45-frame windows.

        Returns:
            (windows, window_is_usable, per_frame_face_detected)
        """
        raw_frames = read_video_frames(context.video_path, frame_skip=1)
        if len(raw_frames) < self.window_size:
            raise InsufficientBufferError(
                f"[{self.MODULE_NAME}] Need at least {self.window_size} frames, got {len(raw_frames)}"
            )

        features: List[List[float]] = []
        face_detected_flags: List[bool] = []
        for frame in raw_frames:
            feature_vector = extract_landmark_features(self.face_landmarker, frame)
            if feature_vector is None:
                face_detected_flags.append(False)
                features.append([0.0] * len(self.feature_names))
            else:
                face_detected_flags.append(True)
                features.append(feature_vector)

        std_safe = np.where(self.normalization_std == 0, 1.0, self.normalization_std)
        normalized = (np.array(features, dtype=np.float64) - self.normalization_mean) / std_safe

        windows: List[np.ndarray] = []
        window_is_usable: List[bool] = []
        for start in range(0, len(normalized) - self.window_size + 1, self.window_size):
            end = start + self.window_size
            windows.append(normalized[start:end])
            window_is_usable.append(all(face_detected_flags[start:end]))

        return windows, window_is_usable, face_detected_flags

    def predict(self, context: InferenceContext) -> PredictionResult:
        """Runs the full pipeline and returns a `PredictionResult` matching the Standard Output Contract."""
        self._ensure_loaded()
        windows, window_is_usable, face_detected_flags = self.preprocess(context)

        usable_windows = [window for window, is_usable in zip(windows, window_is_usable) if is_usable]
        if not usable_windows:
            raise FaceNotDetectedError(
                f"[{self.MODULE_NAME}] Face not detected in one or more frames of every window."
            )

        self._recent_action_history = []
        window_results: List[Dict[str, Any]] = []
        for window in usable_windows:
            window_tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(window_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze(0)
            confidence, predicted_index = torch.max(probabilities, dim=0)
            predicted_action = self.action_labels[int(predicted_index.item())]
            self._recent_action_history.append(predicted_action)
            window_results.append({"predicted_action": predicted_action, "confidence": float(confidence.item())})

        yawn_proportion = self._recent_action_history.count("Yawning") / len(self._recent_action_history)

        if yawn_proportion == 0.0:
            fatigue_state = "Alert"
        elif yawn_proportion < LANDMARK_FATIGUE_THRESHOLD:
            fatigue_state = "Mild Fatigue"
        else:
            fatigue_state = "Drowsy"

        last_window = window_results[-1]
        prediction = fatigue_state if LANDMARK_RELIABILITY_GATE_OPEN else last_window["predicted_action"]
        face_detected_ratio = float(np.mean(face_detected_flags)) if face_detected_flags else 0.0

        return self.build_result(
            prediction=prediction,
            confidence=last_window["confidence"],
            risk_score=yawn_proportion,
            metadata={
                "predicted_action": last_window["predicted_action"],
                "fatigue_state": fatigue_state,
                "fatigue_state_available": LANDMARK_RELIABILITY_GATE_OPEN,
                "yawn_proportion": yawn_proportion,
                "window_size_used": self.window_size,
                "frames_processed": len(face_detected_flags),
                "windows_used": len(usable_windows),
                "windows_skipped": len(windows) - len(usable_windows),
                "face_detected_ratio": face_detected_ratio,
                "timeline": window_results,
                "checkpoint": str(self.checkpoint_path),
            },
        )


    # --- Streaming API (legacy batch predict() unchanged) ---
    def reset_streaming_state(self) -> None:
        from collections import deque
        self._stream_feature_buffer = deque(maxlen=self.window_size)
        self._stream_action_history: List[str] = []
        self._stream_since_infer = 0
        self._stream_latest: Optional[PredictionResult] = None
        self._stream_face_flags = deque(maxlen=self.window_size)

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float) -> Dict[str, Any]:
        self._ensure_loaded()
        if not hasattr(self, "_stream_feature_buffer"):
            self.reset_streaming_state()
        feature_vector = extract_landmark_features(self.face_landmarker, frame_bgr)
        if feature_vector is None:
            self._stream_face_flags.append(False)
            feature_vector = [0.0] * len(self.feature_names)
        else:
            self._stream_face_flags.append(True)
        std_safe = np.where(self.normalization_std == 0, 1.0, self.normalization_std)
        normalized = (np.array(feature_vector, dtype=np.float64) - self.normalization_mean) / std_safe
        self._stream_feature_buffer.append(normalized)
        buffer_len = len(self._stream_feature_buffer)
        if buffer_len < self.window_size:
            return {"status": "WARMING_UP", "result": None, "buffer_len": buffer_len}
        if not all(self._stream_face_flags):
            return {"status": "UNAVAILABLE", "result": self._stream_latest, "buffer_len": buffer_len}
        self._stream_since_infer += 1
        if self._stream_since_infer % LANDMARK_STRIDE != 0 and self._stream_latest is not None:
            return {"status": "READY", "result": self._stream_latest, "buffer_len": buffer_len}
        window = np.stack(list(self._stream_feature_buffer), axis=0)
        window_tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(window_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_index = torch.max(probabilities, dim=0)
        predicted_action = self.action_labels[int(predicted_index.item())]
        self._stream_action_history.append(predicted_action)
        yawn_proportion = self._stream_action_history.count("Yawning") / len(self._stream_action_history)
        if yawn_proportion == 0.0:
            fatigue_state = "Alert"
        elif yawn_proportion < LANDMARK_FATIGUE_THRESHOLD:
            fatigue_state = "Mild Fatigue"
        else:
            fatigue_state = "Drowsy"
        prediction = fatigue_state if LANDMARK_RELIABILITY_GATE_OPEN else predicted_action
        self._stream_latest = self.build_result(
            prediction=prediction,
            confidence=float(confidence.item()),
            risk_score=yawn_proportion,
            metadata={
                "predicted_action": predicted_action,
                "fatigue_state": fatigue_state,
                "yawn_proportion": yawn_proportion,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "buffer_len": buffer_len,
                "mode": "streaming",
            },
        )
        return {"status": "READY", "result": self._stream_latest, "buffer_len": buffer_len}

    def warmup(self) -> None:
        """Loads the model + Face Landmarker (if needed) and runs a single dummy LSTM forward pass."""
        self._ensure_loaded()
        dummy_window = torch.zeros((1, self.window_size, LANDMARK_INPUT_SIZE), device=self.device)
        with torch.no_grad():
            self.model(dummy_window)
        self._logger.info("[%s] Warmup completed", self.MODULE_NAME)

    def unload(self) -> None:
        """Releases the LSTM model, the MediaPipe Face Landmarker, and any device memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.face_landmarker is not None:
            self.face_landmarker.close()
            self.face_landmarker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("LandmarkFatigueAdapter ready.")



# ==========================================================
# Driver Wellness AI — Smoking & Drinking Adapter (YOLOv8n)
#   + TEMPORAL-CONSISTENCY LAYER  (smoking & drinking only)
# ==========================================================
from collections import deque, defaultdict
from ultralytics import YOLO


class TemporalVoter:
    """Sliding-window majority vote + hysteresis, per class.

    Feed it the SET of class names detected in each frame; it returns the set of
    currently CONFIRMED (ON) classes. Short flickers never accumulate enough
    hits to switch ON; a confirmed class is only released once its hit count
    falls to the OFF threshold (hysteresis), which prevents on/off chattering
    and holds genuine sustained actions (keeping false negatives low).
    """

    def __init__(self, window: int, on_hits: int, off_hits: int) -> None:
        self.on_hits, self.off_hits = on_hits, off_hits
        self.hist = defaultdict(lambda: deque(maxlen=window))
        self.state = defaultdict(bool)

    def update(self, present):
        for c in set(self.hist) | set(present):
            self.hist[c].append(1 if c in present else 0)
            hits = sum(self.hist[c])
            if not self.state[c] and hits >= self.on_hits:
                self.state[c] = True
            elif self.state[c] and hits <= self.off_hits:
                self.state[c] = False
        return {c for c, on in self.state.items() if on}

    @staticmethod
    def from_fps(fps, window_sec, on_frac, off_frac):
        """Builds a voter whose window is `window_sec` of real time at `fps`."""
        window = max(1, round(window_sec * fps))
        on_hits = max(1, round(window * on_frac))
        off_hits = max(0, round(window * off_frac))
        return TemporalVoter(window, on_hits, off_hits), window, on_hits, off_hits


# BGR colors for drawing confirmed detections
_SMOKING_COLORS = {"drinking": (0, 0, 255), "smoking": (0, 165, 255)}


def _draw_confirmed(frame, raw, confirmed):
    """Draws ONLY the boxes whose class is currently temporally confirmed,
    plus a status banner. `raw` = list of (class_name, conf, [x1,y1,x2,y2])."""
    for class_name, conf, (x1, y1, x2, y2) in raw:
        if class_name not in confirmed:
            continue  # flicker suppressed -> no box
        color = _SMOKING_COLORS.get(class_name, (0, 200, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    banner = ("ALERT: " + ", ".join(sorted(confirmed))) if confirmed else "normal"
    cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if confirmed else (0, 180, 0), 2)
    return frame


class SmokingAdapter(BaseModelAdapter):
    """
    Inference adapter for the Smoking & Drinking Detection model
    (`Smoking_And_Drinking.pt`, an Ultralytics YOLOv8n checkpoint), with a
    temporal-consistency layer applied to the smoking & drinking classes.
    """

    MODULE_KEY: str = "smoking"
    MODULE_NAME: str = "Smoking & Drinking Detection"

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.imgsz: int = SMOKING_IMGSZ
        self.conf_threshold: float = SMOKING_CONF_THRESHOLD
        # --- temporal-layer settings (from config, with safe fallbacks) ---
        self.conf_per_class = dict(globals().get(
            "SMOKING_CONF_PER_CLASS", {"smoking": 0.25, "drinking": 0.40}))
        self.window_sec = float(globals().get("SMOKING_WINDOW_SEC", 0.50))
        self.on_frac = float(globals().get("SMOKING_ON_FRAC", 0.50))
        self.off_frac = float(globals().get("SMOKING_OFF_FRAC", 0.30))

    def load_model(self) -> None:
        """Loads the YOLOv8n checkpoint via `ultralytics.YOLO` (never `torch.load`)."""
        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"[{self.MODULE_NAME}] Checkpoint not found: {self.checkpoint_path}")

        self._logger.info("[%s] Loading YOLO checkpoint from %s", self.MODULE_NAME, self.checkpoint_path)
        try:
            model = YOLO(str(self.checkpoint_path))
        except Exception as exc:  # noqa: BLE001 - Ultralytics raises broadly on bad checkpoints
            raise AdapterError(f"[{self.MODULE_NAME}] Failed to load YOLO checkpoint: {exc}") from exc

        self.model = model
        self._logger.info("[%s] YOLO model loaded. Classes: %s", self.MODULE_NAME, model.names)

    def preprocess(self, context: InferenceContext) -> str:
        """Validates the video is decodable and returns its resolved path."""
        video_path = Path(context.video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecodeError(f"[{self.MODULE_NAME}] Cannot decode video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frame_count <= 0:
            raise InsufficientBufferError(f"[{self.MODULE_NAME}] Video reports zero frames: {video_path}")
        return str(video_path)

    def _detect_frame(self, frame, device_arg):
        """Raw per-frame YOLO detection, filtered by the per-class conf floor.
        Returns a list of (class_name, confidence, [x1,y1,x2,y2], class_id)."""
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf_threshold,
                                 device=device_arg, verbose=False)[0]
        raw = []
        if res.boxes is not None:
            for box in res.boxes:
                class_id = int(box.cls.item())
                names = res.names
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                confidence = float(box.conf.item())
                if confidence < self.conf_per_class.get(class_name, self.conf_threshold):
                    continue  # below this class's floor -> ignore
                box_xyxy = [float(v) for v in box.xyxy.squeeze(0).tolist()]
                raw.append((class_name, confidence, box_xyxy, class_id))
        return raw

    def predict(self, context: InferenceContext) -> PredictionResult:
        """Frame-by-frame YOLO detection with a temporal-consistency vote.
        Only CONFIRMED detections feed the aggregated risk / boxes / classes."""
        self._ensure_loaded()
        resolved_path = self.preprocess(context)
        # Per spec Section 3: device is selected at predict-time, not the constructor.
        device_arg = 0 if self.device.type == "cuda" else "cpu"

        cap = cv2.VideoCapture(resolved_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        voter, window, on_hits, off_hits = TemporalVoter.from_fps(
            fps, self.window_sec, self.on_frac, self.off_frac)
        self._logger.info("[%s] temporal layer: fps=%.1f window=%df on>=%d off<=%d",
                          self.MODULE_NAME, fps, window, on_hits, off_hits)

        # Optional live inline stream (no file written, no download). Off by
        # default so the end-to-end pipeline stays silent; the dedicated live
        # cell sets context.extra["smoking_live_stream"]=True.
        live = bool(context.extra.get("smoking_live_stream", False))
        if live:
            from IPython.display import display as _ipy_display, Image as _IPyImage, clear_output

        detections: List[Dict[str, Any]] = []
        timeline: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        classes_detected = set()
        max_confidence = 0.0
        frames_processed = 0
        prev_confirmed = set()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = frames_processed
            frames_processed += 1

            raw = self._detect_frame(frame, device_arg)         # per-frame, per-class conf
            present = {r[0] for r in raw}
            confirmed = voter.update(present)                   # temporal vote + hysteresis

            # log ON/OFF transitions
            for c in confirmed - prev_confirmed:
                events.append({"frame_index": frame_index, "time_sec": frame_index / fps,
                               "class_name": c, "event": "START"})
            for c in prev_confirmed - confirmed:
                events.append({"frame_index": frame_index, "time_sec": frame_index / fps,
                               "class_name": c, "event": "END"})
            prev_confirmed = confirmed

            # keep ONLY confirmed detections for the aggregate result
            frame_confirmed = []
            for class_name, confidence, box_xyxy, class_id in raw:
                if class_name not in confirmed:
                    continue
                detections.append({
                    "frame_index": frame_index,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box_xyxy": box_xyxy,
                })
                classes_detected.add(class_name)
                max_confidence = max(max_confidence, confidence)
                frame_confirmed.append(class_name)

            timeline.append({
                "frame_index": frame_index,
                "classes": frame_confirmed,                 # confirmed (drawn) classes
                "raw_classes": sorted(present),             # before smoothing
                "num_detections": len(frame_confirmed),
            })

            if live:
                annotated = _draw_confirmed(frame, [(r[0], r[1], r[2]) for r in raw], confirmed)
                ok_enc, buf = cv2.imencode(".jpg", annotated)
                if ok_enc:
                    clear_output(wait=True)
                    _ipy_display(_IPyImage(data=buf.tobytes()))

        cap.release()

        classes_detected_sorted = sorted(classes_detected)

        if not classes_detected_sorted:
            prediction, confidence, risk_score = "none", 0.0, 0.0
        elif len(classes_detected_sorted) > 1:
            prediction = "smoking+drinking" if set(classes_detected_sorted) == {"smoking", "drinking"} else "+".join(classes_detected_sorted)
            confidence = max_confidence
            risk_score = min(1.0, SMOKING_BOTH_DETECTED_RISK * confidence)
        else:
            prediction = classes_detected_sorted[0]
            confidence = max_confidence
            base_risk = SMOKING_RISK_WEIGHTS.get(prediction, 1.0)
            risk_score = min(1.0, base_risk * confidence)

        bounding_boxes = [
            {
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "x1": detection["box_xyxy"][0],
                "y1": detection["box_xyxy"][1],
                "x2": detection["box_xyxy"][2],
                "y2": detection["box_xyxy"][3],
            }
            for detection in detections
        ]

        return self.build_result(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "bounding_boxes": bounding_boxes,
                "detections": detections,
                "num_detections": len(detections),
                "classes_detected": classes_detected_sorted,
                "frame_number": None,
                "frames_processed": frames_processed,
                "conf_threshold": self.conf_threshold,
                "imgsz": self.imgsz,
                "timeline": timeline,
                "events": events,
                "temporal": {
                    "fps": fps,
                    "window_frames": window,
                    "on_hits": on_hits,
                    "off_hits": off_hits,
                    "window_sec": self.window_sec,
                    "on_frac": self.on_frac,
                    "off_frac": self.off_frac,
                    "conf_per_class": self.conf_per_class,
                },
                "model_version": self.checkpoint_path.stem,
                "checkpoint": str(self.checkpoint_path),
            },
        )


    # --- Streaming API (TemporalVoter persists across entire video/session) ---
    def reset_streaming_state(self, fps: float = 25.0) -> None:
        self._stream_fps = fps
        self._voter, self._window, self._on_hits, self._off_hits = TemporalVoter.from_fps(
            fps, self.window_sec, self.on_frac, self.off_frac
        )
        self._prev_confirmed = set()
        self._stream_events: List[Dict[str, Any]] = []
        self._stream_detections: List[Dict[str, Any]] = []
        self._stream_classes = set()
        self._stream_max_conf = 0.0
        self._stream_latest: Optional[PredictionResult] = None
        self._stream_last_raw: List[tuple] = []
        self._stream_last_confirmed: set = set()

    def _build_stream_aggregate(self, frames_processed: int) -> PredictionResult:
        classes_detected_sorted = sorted(self._stream_classes)
        if not classes_detected_sorted:
            prediction, confidence, risk_score = "none", 0.0, 0.0
        elif len(classes_detected_sorted) > 1:
            prediction = (
                "smoking+drinking"
                if set(classes_detected_sorted) == {"smoking", "drinking"}
                else "+".join(classes_detected_sorted)
            )
            confidence = self._stream_max_conf
            risk_score = min(1.0, SMOKING_BOTH_DETECTED_RISK * confidence)
        else:
            prediction = classes_detected_sorted[0]
            confidence = self._stream_max_conf
            base_risk = SMOKING_RISK_WEIGHTS.get(prediction, 1.0)
            risk_score = min(1.0, base_risk * confidence)
        return self.build_result(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "classes_detected": classes_detected_sorted,
                "frames_processed": frames_processed,
                "events": list(self._stream_events),
                "confirmed_now": sorted(self._stream_last_confirmed),
                "raw_classes": sorted({r[0] for r in self._stream_last_raw}),
                "mode": "streaming",
                "temporal": {
                    "fps": self._stream_fps,
                    "window_frames": self._window,
                    "on_hits": self._on_hits,
                    "off_hits": self._off_hits,
                },
            },
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        if not hasattr(self, "_voter"):
            self.reset_streaming_state(fps=fps or 25.0)
        if frame_index % SMOKING_FRAME_STRIDE != 0 and self._stream_latest is not None:
            return {
                "status": "READY",
                "result": self._stream_latest,
                "events": [],
                "annotated_frame": _draw_confirmed(
                    frame_bgr.copy(),
                    [(r[0], r[1], r[2]) for r in self._stream_last_raw],
                    self._stream_last_confirmed,
                ),
            }
        device_arg = 0 if self.device.type == "cuda" else "cpu"
        raw = self._detect_frame(frame_bgr, device_arg)
        present = {r[0] for r in raw}
        confirmed = self._voter.update(present)
        new_events = []
        for class_name in confirmed - self._prev_confirmed:
            evt = {"frame_index": frame_index, "time_sec": timestamp_sec, "class_name": class_name, "event": "START"}
            self._stream_events.append(evt)
            new_events.append(evt)
        for class_name in self._prev_confirmed - confirmed:
            evt = {"frame_index": frame_index, "time_sec": timestamp_sec, "class_name": class_name, "event": "END"}
            self._stream_events.append(evt)
            new_events.append(evt)
        self._prev_confirmed = confirmed
        for class_name, confidence, box_xyxy, class_id in raw:
            if class_name not in confirmed:
                continue
            self._stream_detections.append({
                "frame_index": frame_index,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box_xyxy": box_xyxy,
            })
            self._stream_classes.add(class_name)
            self._stream_max_conf = max(self._stream_max_conf, confidence)
        self._stream_last_raw = raw
        self._stream_last_confirmed = confirmed
        self._stream_latest = self._build_stream_aggregate(frame_index + 1)
        annotated = _draw_confirmed(
            frame_bgr.copy(),
            [(r[0], r[1], r[2]) for r in raw],
            confirmed,
        )
        return {
            "status": "READY",
            "result": self._stream_latest,
            "events": new_events,
            "annotated_frame": annotated,
        }

    def warmup(self) -> None:
        """Loads the model (if needed) and runs one dummy prediction on a blank 640x640 image."""
        self._ensure_loaded()
        dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        device_arg = 0 if self.device.type == "cuda" else "cpu"
        self.model.predict(source=dummy_frame, imgsz=self.imgsz, device=device_arg, verbose=False)
        self._logger.info("[%s] Warmup completed", self.MODULE_NAME)

    def unload(self) -> None:
        """Releases the model and frees device memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("SmokingAdapter ready (temporal-consistency layer active for smoking & drinking).")



import time
import gc
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from collections import deque, defaultdict
from ultralytics import YOLO

class SequentialConsensusFilter:
    """
    A stateful temporal consensus engine utilizing sliding-window frequency
    analysis and hysteresis damping to stabilize object detections across video streams.
    """
    def __init__(self, time_window_frames: int, activation_ratio: float, release_ratio: float) -> None:
        self.window_capacity = time_window_frames
        self.activate_limit = activation_ratio
        self.release_limit = release_ratio
        self.frame_history = defaultdict(lambda: deque(maxlen=time_window_frames))
        self.active_signals = defaultdict(bool)

    def evaluate(self, detected_classes: set) -> set:
        observation_space = set(self.frame_history.keys()) | set(detected_classes)
        for class_tag in observation_space:
            self.frame_history[class_tag].append(1 if class_tag in detected_classes else 0)
            accumulated_score = sum(self.frame_history[class_tag])
            current_ratio = accumulated_score / float(self.window_capacity)

            if not self.active_signals[class_tag] and current_ratio >= self.activate_limit:
                self.active_signals[class_tag] = True
            elif self.active_signals[class_tag] and current_ratio <= self.release_limit:
                self.active_signals[class_tag] = False

        return {class_tag for class_tag, is_active in self.active_signals.items() if is_active}

    @classmethod
    def initialize_from_video_stream(cls, fps: float, duration_sec: float, activate_pct: float, release_pct: float):
        frame_span = max(1, round(duration_sec * fps))
        activate_threshold = max(1, round(frame_span * activate_pct))
        release_threshold = max(0, round(frame_span * release_pct))
        return cls(frame_span, activate_threshold / frame_span, release_threshold / frame_span), frame_span


class SeatBeltPhoneDetectionAdapter(BaseModelAdapter):
    """
    Inference adapter for the Seat Belt & Phone Usage Detection model.
    Optimized with 1280px upscaling, Gamma shadow correction, and class-agnostic NMS.
    """

    MODULE_KEY: str = "seatbelt"
    MODULE_NAME: str = "Seat Belt & Phone Usage Detection"

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.imgsz: int = 1280  # Upgraded to 1280 for high-res micro-feature detection
        self.conf_threshold: float = SEATBELT_PHONE_CONF_THRESHOLD
        self.risk_map: Dict[str, float] = SEATBELT_PHONE_RISK_MAP
        # Use SEATBELT_CONF_PER_CLASS from config, which includes Phone: 0.02, Seatbelt: 0.35
        self.class_floors = SEATBELT_CONF_PER_CLASS
        self.window_sec = float(globals().get("SEATBELT_WINDOW_SEC", 0.50))
        self.on_frac = float(globals().get("SEATBELT_ON_FRAC", 0.40)) # Use configured ON_FRAC
        self.off_frac = float(globals().get("SEATBELT_OFF_FRAC", 0.20)) # Use configured OFF_FRAC

    def load_model(self) -> None:
        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"[{self.MODULE_NAME}] Checkpoint not found: {self.checkpoint_path}")

        self._logger.info("[%s] Loading YOLO checkpoint from %s", self.MODULE_NAME, self.checkpoint_path)
        try:
            model = YOLO(str(self.checkpoint_path))
        except Exception as exc:
            raise AdapterError(f"[{self.MODULE_NAME}] Failed to load YOLO checkpoint: {exc}") from exc

        self.model = model
        self._logger.info("[%s] YOLO model loaded. Classes: %s", self.MODULE_NAME, model.names)

    def preprocess(self, context: InferenceContext) -> str:
        video_path = Path(context.video_path)
        if not video_path.is_file():
            raise VideoDecodeError(f"[{self.MODULE_NAME}] Cannot find video: {video_path}")
        return str(video_path)

    def _detect_frame(self, frame, device_arg):
        """Raw per-frame YOLO detection, filtered by the per-class conf floor.
        Returns a list of (class_name, confidence, [x1,y1,x2,y2], class_id)."""
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=0.10, # Initial low confidence for raw detections
            device=device_arg,
            verbose=False
        )
        res = results[0]
        raw = []
        if res.boxes is not None:
            for box in res.boxes:
                class_id = int(box.cls.item())
                raw_name = res.names.get(class_id, str(class_id)) if isinstance(res.names, dict) else str(class_id)
                confidence = float(box.conf.item())

                # Map raw YOLO classes to standardized names
                if "phone" in raw_name.lower():
                    class_name = "Phone"
                elif "seat" in raw_name.lower() or "belt" in raw_name.lower():
                    class_name = "Seatbelt"
                else:
                    class_name = raw_name.capitalize() # Fallback for unknown classes, though not expected

                # Apply per-class confidence floors from configuration
                if confidence < self.class_floors.get(class_name, self.conf_threshold):
                    continue # Below this class's floor -> ignore

                x1, y1, x2, y2 = [float(v) for v in box.xyxy.squeeze(0).tolist()]

                # --- SPATIAL FILTER: REJECT WINDOW REFLECTIONS FOR SEATBELT ---
                # This logic is adapted from the predict method to be applied per frame
                h, w, _ = frame.shape
                if class_name == "Seatbelt":
                    # If the detection is trapped entirely in the top-left/top-right window glass zone:
                    if y2 < (h * 0.35) and (x1 < (w * 0.25) or x2 > (w * 0.75)):
                        continue # Ignore the window reflection!

                raw.append((class_name, confidence, [x1,y1,x2,y2], class_id))
        return raw

    def predict(self, context: InferenceContext) -> PredictionResult:
        self._ensure_loaded()
        resolved_path = self.preprocess(context)
        device_arg = 0 if self.device.type == "cuda" else "cpu"

        start_time = time.time()
        capture = cv2.VideoCapture(resolved_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

        consensus_engine, frame_capacity = SequentialConsensusFilter.initialize_from_video_stream(
            fps, self.window_sec, self.on_frac, self.off_frac
        )

        detections: List[Dict[str, Any]] = []
        timeline: List[Dict[str, Any]] = []
        best_confidence_by_class: Dict[str, float] = {}
        frames_processed = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frames_processed += 1

            h, w, _ = frame.shape

            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=0.10,
                device=device_arg,
                verbose=False
            )

            frame_raw_detections = []
            frame_observed_classes = set()

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    raw_name = results[0].names.get(class_id, str(class_id)) if isinstance(results[0].names, dict) else str(class_id)
                    confidence = float(box.conf.item())

                    if "phone" in raw_name.lower():
                        class_name = "Phone"
                    elif "seat" in raw_name.lower() or "belt" in raw_name.lower():
                        class_name = "Seatbelt"
                    else:
                        class_name = raw_name.capitalize()

                    x1, y1, x2, y2 = [float(v) for v in box.xyxy.squeeze(0).tolist()]

                    # --- SPATIAL FILTER: REJECT WINDOW REFLECTIONS ---
                    # Real seatbelts cross the torso (middle/lower region of the cabin box).
                    # Window glass reflections appear on the extreme upper edges or corners.
                    if class_name == "Seatbelt":
                        # If the detection is trapped entirely in the top-left/top-right window glass zone:
                        if y2 < (h * 0.35) and (x1 < (w * 0.25) or x2 > (w * 0.75)):
                            continue # Ignore the window reflection!
                        if confidence < 0.65:
                            continue

                    if class_name == "Phone" and confidence < 0.05:
                        continue

                    frame_raw_detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    })
                    frame_observed_classes.add(class_name)

            stabilized_classes = consensus_engine.evaluate(frame_observed_classes)

            frame_confirmed_names = []
            for det in frame_raw_detections:
                c_name = det["class"]
                if c_name in stabilized_classes:
                    detections.append(det)
                    best_confidence_by_class[c_name] = max(best_confidence_by_class.get(c_name, 0.0), det["confidence"])
                    frame_confirmed_names.append(c_name)

            timeline.append({
                "frame_index": frames_processed - 1,
                "classes": frame_confirmed_names,
                "num_detections": len(frame_confirmed_names)
            })

        capture.release()
        inference_time_ms = (time.time() - start_time) * 1000.0

        has_phone = "Phone" in best_confidence_by_class
        has_seatbelt = "Seatbelt" in best_confidence_by_class

        if has_phone and has_seatbelt:
            prediction = "Phone & Seatbelt"
            confidence = max(best_confidence_by_class["Phone"], best_confidence_by_class["Seatbelt"])
        elif has_phone:
            prediction = "Phone Only"
            confidence = best_confidence_by_class["Phone"]
        elif has_seatbelt:
            prediction = "Seatbelt Only"
            confidence = best_confidence_by_class["Seatbelt"]
        else:
            prediction = "No Detection"
            confidence = 0.0

        risk_score = self.risk_map.get(prediction, 0.00)

        return self.build_result(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "detections": detections,
                "inference_time_ms": inference_time_ms,
                "num_detections": len(detections),
                "frames_processed": frames_processed,
                "conf_threshold": self.conf_threshold,
                "imgsz": self.imgsz,
                "timeline": timeline,
                "model_version": self.checkpoint_path.stem,
                "checkpoint": str(self.checkpoint_path),
            },
        )

    # --- Streaming API (TemporalVoter persists across entire video/session) ---
    def reset_streaming_state(self, fps: float = 25.0) -> None:
        self._stream_fps = fps
        self._consensus_engine, self._window_frames = SequentialConsensusFilter.initialize_from_video_stream(
            fps, self.window_sec, self.on_frac, self.off_frac
        )
        self._prev_confirmed = set()
        self._stream_events: List[Dict[str, Any]] = []
        self._stream_detections_history: List[Dict[str, Any]] = [] # For storing raw detections of confirmed classes
        self._stream_classes_history = set()
        self._stream_max_conf = 0.0
        self._stream_latest: Optional[PredictionResult] = None
        self._stream_last_raw_detections: List[tuple] = [] # Stores (class_name, conf, box_xyxy) for current frame
        self._stream_last_confirmed_classes: set = set()

    def _build_stream_aggregate(self, frames_processed: int) -> PredictionResult:
        has_phone = "Phone" in self._stream_classes_history
        has_seatbelt = "Seatbelt" in self._stream_classes_history

        # Determine prediction and overall confidence based on history
        if has_phone and has_seatbelt:
            prediction = "Phone & Seatbelt"
            # For streaming, we might not have a single `best_confidence_by_class` easily,
            # so we'll use the highest confidence observed for any of the detected classes.
            # Alternatively, if we store per-class max conf in streaming, we can use that.
            # For now, let's use the current frame's max conf for the reported prediction.
            # This can be refined based on how `_stream_max_conf` is updated.
            confidence = self._stream_max_conf # This should be the max conf of *any* detected object that led to the current state
        elif has_phone:
            prediction = "Phone Only"
            confidence = self._stream_max_conf
        elif has_seatbelt:
            prediction = "Seatbelt Only"
            confidence = self._stream_max_conf
        else:
            prediction = "No Detection"
            confidence = 0.0

        risk_score = self.risk_map.get(prediction, 0.00)

        return self.build_result(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "classes_detected": sorted(list(self._stream_classes_history)),
                "frames_processed": frames_processed,
                "events": list(self._stream_events),
                "confirmed_now": sorted(list(self._stream_last_confirmed_classes)),
                "raw_classes_current_frame": sorted({r[0] for r in self._stream_last_raw_detections}),
                "mode": "streaming",
                "temporal": {
                    "fps": self._stream_fps,
                    "window_frames": self._window_frames,
                    "window_sec": self.window_sec,
                    "on_frac": self.on_frac,
                    "off_frac": self.off_frac,
                },
            },
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        if not hasattr(self, "_consensus_engine"):
            self.reset_streaming_state(fps=fps or 25.0)

        # Only process every SEATBELT_PHONE_FRAME_STRIDE frames
        if frame_index % SEATBELT_PHONE_FRAME_STRIDE != 0 and self._stream_latest is not None:
            # If not processing this frame, return the last known state and an annotated frame if possible
            # For Seatbelt, we might want to return a copy of the original frame if there's no annotation to apply
            return {
                "status": "READY",
                "result": self._stream_latest,
                "events": [],
                "annotated_frame": frame_bgr.copy(), # Return original frame if not processing for efficiency
            }

        device_arg = 0 if self.device.type == "cuda" else "cpu"
        raw_detections_info = self._detect_frame(frame_bgr, device_arg)
        present_classes = {r[0] for r in raw_detections_info}

        confirmed_classes = self._consensus_engine.evaluate(present_classes)

        new_events = []
        for class_name in confirmed_classes - self._prev_confirmed:
            evt = {"frame_index": frame_index, "time_sec": timestamp_sec, "class_name": class_name, "event": "START"}
            self._stream_events.append(evt)
            new_events.append(evt)
        for class_name in self._prev_confirmed - confirmed_classes:
            evt = {"frame_index": frame_index, "time_sec": timestamp_sec, "class_name": class_name, "event": "END"}
            self._stream_events.append(evt)
            new_events.append(evt)
        self._prev_confirmed = confirmed_classes

        # Update streaming history for aggregate result
        current_frame_max_conf = 0.0
        for class_name, confidence, box_xyxy, class_id in raw_detections_info:
            if class_name in confirmed_classes:
                # Only add confirmed detections to the history
                self._stream_detections_history.append({
                    "frame_index": frame_index,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box_xyxy": box_xyxy,
                })
                self._stream_classes_history.add(class_name)
                current_frame_max_conf = max(current_frame_max_conf, confidence)

        # Update the overall max confidence for the aggregate result
        if current_frame_max_conf > 0: # Only update if there were detections in this frame
            self._stream_max_conf = max(self._stream_max_conf, current_frame_max_conf)

        self._stream_last_raw_detections = [(r[0], r[1], r[2]) for r in raw_detections_info] # Store (class_name, conf, box_xyxy) for drawing
        self._stream_last_confirmed_classes = confirmed_classes
        self._stream_latest = self._build_stream_aggregate(frame_index + 1)

        # Prepare annotated frame for display
        annotated_frame = frame_bgr.copy()
        h, w, _ = annotated_frame.shape
        for class_name, confidence, box_xyxy, _ in raw_detections_info:
            if class_name in confirmed_classes:
                x1, y1, x2, y2 = [int(v) for v in box_xyxy]
                color = (0, 255, 0) # Green for confirmed detections
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display overall prediction/status on the frame
        if self._stream_latest:
            status_text = f"Risk: {self._stream_latest.prediction} (Score: {self._stream_latest.risk_score:.2f})"
            cv2.putText(annotated_frame, status_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text

        return {
            "status": "READY",
            "result": self._stream_latest,
            "events": new_events,
            "annotated_frame": annotated_frame,
        }

    def warmup(self) -> None:
        self._ensure_loaded()
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        device_arg = 0 if self.device.type == "cuda" else "cpu"
        self.model.predict(source=dummy_frame, imgsz=640, device=device_arg, verbose=False)
        self._logger.info("[%s] Warmup completed", self.MODULE_NAME)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



# ==========================================================
# Driver Wellness AI — Model Registry
# ==========================================================
import os

import pandas as pd
# (IPython display import removed for Spaces)

MODEL_REGISTRY = {
    "video_fatigue": {
        "name": "🎥 Video Fatigue Detection",
        "checkpoint": str(MODEL_ROOT / "Video_Fatigue.pth"),
        "weight": RISK_FUSION_WEIGHTS["video_fatigue"],
    },
    "landmark_fatigue": {
        "name": "😊 Landmark Fatigue Detection",
        "checkpoint": str(MODEL_ROOT / "Landmark_Fatigue.pt"),
        "weight": RISK_FUSION_WEIGHTS["landmark_fatigue"],
    },
    "driver_activity": {
        "name": "🚘 Driver Activity Recognition",
        "checkpoint": str(MODEL_ROOT / "Driver_Activity.pth"),
        "weight": RISK_FUSION_WEIGHTS["driver_activity"],
    },
    "smoking": {
        "name": "🚬 Smoking & Drinking Detection",
        "checkpoint": str(MODEL_ROOT / "Smoking_And_Drinking.pt"),
        "weight": RISK_FUSION_WEIGHTS["smoking"],
    },
    "seatbelt": {
        "name": "📱 Seat Belt & Phone Usage Detection",
        "checkpoint": str(MODEL_ROOT / "SeatBelt_And_Phone.pt"),
        "weight": RISK_FUSION_WEIGHTS["seatbelt"],
    },
}

def _is_git_lfs_pointer(path: str) -> bool:
    """
    True if `path` is an unresolved Git LFS pointer stub rather than the real
    checkpoint (happens when a host's checkout doesn't smudge LFS files) --
    catches the exact failure mode that made os.path.exists() alone report
    "Available" for checkpoints that actually fail to load.
    """
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


system_status = []
available_models = []

print("=" * 80)
print("Driver Wellness AI - Model Availability")
print("=" * 80)

for module_key, module in MODEL_REGISTRY.items():
    exists = os.path.exists(module["checkpoint"]) and not _is_git_lfs_pointer(module["checkpoint"])
    status = "Available" if exists else "Missing"
    icon = "✅" if exists else "❌"

    print(f"{icon} {module['name']}")

    if exists:
        available_models.append(module_key)

    system_status.append(
        {
            "Module": module["name"],
            "Checkpoint": os.path.basename(module["checkpoint"]),
            "Weight": module["weight"],
            "Status": status,
        }
    )

print()
print(f"Available Models : {len(available_models)}/{len(MODEL_REGISTRY)}")

# (table display omitted in module context)

def resolve_checkpoint(module_key: str, filename: str) -> Path:
    """Resolve a checkpoint from MODEL_ROOT, Drive, or extracted ZIP layouts."""
    candidates = [
        MODEL_ROOT / filename,
        Path("/content/models_extracted/Models") / filename,
        Path("/content/drive/MyDrive/Models") / filename,
        Path("/content/Models") / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return MODEL_ROOT / filename


for _module_key, _module in MODEL_REGISTRY.items():
    _module["checkpoint"] = str(resolve_checkpoint(_module_key, Path(_module["checkpoint"]).name))



# ==========================================================
# Driver Wellness AI — Risk Fusion Engine
# ==========================================================
class DriverWellnessRiskFusion:
    """
    Combines `PredictionResult` objects from every module into one overall
    Driver Wellness Score using the Common Driver Risk Score Framework.

    Option A (PDF-pure):
        R_i = severity_w_i × confidence  (only when prediction is risky)
        overall_score = 100 × (1 - exp(-k × R_total))

    Consumes ONLY `PredictionResult` objects — no adapter, checkpoint, or
    architecture knowledge lives here.
    """

    #: (upper_bound_inclusive, label) — the first bucket the score falls into wins.
    RISK_LEVELS: List[Tuple[float, str]] = [
        (25.0, "Low Risk"),
        (50.0, "Moderate Risk"),
        (75.0, "High Risk"),
        (100.0, "Critical Risk"),
    ]

    def __init__(
        self,
        severity_map: Optional[Dict[str, Dict[str, float]]] = None,
        safe_predictions: Optional[Dict[str, set]] = None,
        fusion_k: float = FUSION_EXPONENTIAL_K,
    ) -> None:
        self.severity_map: Dict[str, Dict[str, float]] = (
            dict(severity_map) if severity_map else dict(RISK_EVENT_SEVERITY)
        )
        self.safe_predictions: Dict[str, set] = (
            dict(safe_predictions) if safe_predictions else dict(RISK_EVENT_SAFE_PREDICTIONS)
        )
        self.fusion_k = float(fusion_k)
        # Kept for backwards-compat with older callers that referenced .weights
        self.weights: Dict[str, float] = dict(RISK_FUSION_WEIGHTS)

    def _resolve_module_key(self, result: PredictionResult) -> Optional[str]:
        meta = result.metadata or {}
        module_key = meta.get("module_key")
        if module_key:
            return str(module_key)
        return FUSION_MODULE_NAME_TO_KEY.get(result.module)

    def _severity_for_prediction(self, module_key: str, prediction: str) -> float:
        safe_labels = self.safe_predictions.get(module_key, set())
        if prediction in safe_labels:
            return 0.0
        return float(self.severity_map.get(module_key, {}).get(prediction, 0.0))

    def _event_risk(self, result: PredictionResult) -> Tuple[Optional[str], float, float]:
        """Returns (module_key, severity_w, R_i). R_i is 0 when unavailable or safe."""
        if result.is_error:
            return None, 0.0, 0.0

        module_key = self._resolve_module_key(result)
        if module_key is None:
            return None, 0.0, 0.0

        severity_w = self._severity_for_prediction(module_key, str(result.prediction))
        if severity_w <= 0.0:
            return module_key, 0.0, 0.0

        confidence = max(0.0, min(1.0, float(result.confidence)))
        event_risk = severity_w * confidence
        return module_key, severity_w, event_risk

    def _risk_level(self, score: float) -> str:
        """Maps a 0-100 score to a categorical risk level."""
        for upper_bound, label in self.RISK_LEVELS:
            if score <= upper_bound:
                return label
        return self.RISK_LEVELS[-1][1]

    def fuse(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """
        Args:
            results: One `PredictionResult` per module.

        Returns:
            {
                "overall_score": float in [0, 100],
                "overall_score_raw": same as overall_score before orchestrator smoothing,
                "risk_level": str,
                "r_total": float,
                "fusion_k": float,
                "contributions": [...],
            }
        """
        contributions: List[Dict[str, Any]] = []
        r_total = 0.0

        for result in results:
            module_key, severity_w, event_risk = self._event_risk(result)
            is_available = not result.is_error
            if is_available:
                r_total += event_risk

            contributions.append(
                {
                    "module": result.module,
                    "module_key": module_key,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "risk_score": result.risk_score,
                    "severity_weight": severity_w,
                    "event_risk": round(event_risk, 4),
                    "weight": result.weight,
                    "available": is_available,
                    "weighted_contribution": round(event_risk, 2),
                }
            )

        if r_total > 0.0:
            overall_score = round(100.0 * (1.0 - math.exp(-self.fusion_k * r_total)), 2)
        else:
            overall_score = 0.0

        risk_level = self._risk_level(overall_score)

        return {
            "overall_score": overall_score,
            "overall_score_raw": overall_score,
            "risk_level": risk_level,
            "r_total": round(r_total, 4),
            "fusion_k": self.fusion_k,
            "contributions": contributions,
        }

    def fuse_streaming_states(self, module_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse latest streaming module states; WARMING_UP/UNAVAILABLE modules are excluded."""
        ready_results: List[PredictionResult] = []
        module_status = {}
        for module_key, state in module_states.items():
            status = state.get("status", "UNAVAILABLE")
            module_status[module_key] = status
            if status == "READY" and state.get("result") is not None:
                result = state["result"]
                result.weight = RISK_FUSION_WEIGHTS.get(module_key, 0.0)
                result.metadata = dict(result.metadata or {})
                result.metadata["module_key"] = module_key
                ready_results.append(result)
        fusion = self.fuse(ready_results)
        fusion["module_status"] = module_status
        fusion["modules_ready"] = [k for k, s in module_status.items() if s == "READY"]
        fusion["modules_warming_up"] = [k for k, s in module_status.items() if s == "WARMING_UP"]
        fusion["modules_unavailable"] = [k for k, s in module_status.items() if s not in ("READY", "WARMING_UP")]
        return fusion


risk_fusion_engine = DriverWellnessRiskFusion()
print("DriverWellnessRiskFusion ready (Option A exponential fusion).")



# ==========================================================
# Streaming Wellness Orchestrator
# Single-pass decode · bounded buffers · continuous risk fusion
# ==========================================================
import json
import time
# psutil imported (guarded) in header
from collections import deque

# IPython imported (guarded) in header


# tqdm imported in header


def _ram_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _gpu_mem_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
    }


class StreamingWellnessOrchestrator:
    """Single-pass streaming orchestrator for TA sliding-window simulation."""

    MODULE_KEYS = [
        "video_fatigue",
        "landmark_fatigue",
        "driver_activity",
        "smoking",
        "seatbelt",
    ]

    def __init__(
        self,
        manager: DriverWellnessModuleManager,
        fusion_engine: DriverWellnessRiskFusion,
    ) -> None:
        self.manager = manager
        self.fusion_engine = fusion_engine
        self.latest_module_states: Dict[str, Dict[str, Any]] = {}
        self.latest_fusion: Optional[Dict[str, Any]] = None
        self.segment_summaries: List[Dict[str, Any]] = []
        self.all_events: List[Dict[str, Any]] = []
        self.score_timeline: deque = deque(maxlen=MAX_TIMELINE_ENTRIES)
        self.module_timeline: Dict[str, List[Dict[str, Any]]] = {key: [] for key in self.MODULE_KEYS}
        self._source_fps = 25.0
        self._last_fusion_ts = -1.0
        self._next_segment_end = SEGMENT_SUMMARY_SEC
        self._last_display_ts = -1.0
        self._wall_start = 0.0
        self._frames_processed = 0
        self._peak_ram_mb = 0.0
        self._initial_ram_mb = _ram_mb()
        self._score_smooth_buffer: deque = deque(maxlen=500)

    def _apply_fusion_smoothing(self, fusion: Dict[str, Any], timestamp_sec: float) -> Dict[str, Any]:
        """PDF Step 5: moving average of raw exponential score over a short window."""
        raw_score = float(fusion.get("overall_score_raw", fusion["overall_score"]))
        fusion["overall_score_raw"] = round(raw_score, 2)
        if not FUSION_SCORE_SMOOTHING_ENABLED:
            fusion["overall_score"] = round(raw_score, 2)
            fusion["risk_level"] = self.fusion_engine._risk_level(fusion["overall_score"])
            return fusion

        self._score_smooth_buffer.append((timestamp_sec, raw_score))
        cutoff = timestamp_sec - FUSION_SCORE_SMOOTHING_WINDOW_SEC
        window_scores = [score for ts, score in self._score_smooth_buffer if ts >= cutoff]
        smoothed = sum(window_scores) / len(window_scores) if window_scores else raw_score
        fusion["overall_score"] = round(smoothed, 2)
        fusion["risk_level"] = self.fusion_engine._risk_level(fusion["overall_score"])
        return fusion

    def reset(self, fps: float = 25.0) -> None:
        self._source_fps = fps or 25.0
        for module_key in self.MODULE_KEYS:
            adapter = self.manager.get_adapter(module_key)
            if hasattr(adapter, "reset_streaming_state"):
                if module_key == "smoking":
                    adapter.reset_streaming_state(fps=self._source_fps)
                else:
                    adapter.reset_streaming_state()
        self.latest_module_states = {}
        self.latest_fusion = None
        self.segment_summaries = []
        self.all_events = []
        self.score_timeline = deque(maxlen=MAX_TIMELINE_ENTRIES)
        self.module_timeline = {key: [] for key in self.MODULE_KEYS}
        self._last_fusion_ts = -1.0
        self._next_segment_end = SEGMENT_SUMMARY_SEC
        self._last_display_ts = -1.0
        self._frames_processed = 0
        self._peak_ram_mb = _ram_mb()
        self._initial_ram_mb = self._peak_ram_mb
        self._score_smooth_buffer.clear()
        RECOVERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float) -> Dict[str, Any]:
        for module_key in self.MODULE_KEYS:
            adapter = self.manager.get_adapter(module_key)
            if module_key == "smoking":
                state = adapter.process_frame(frame_bgr, frame_index, timestamp_sec, fps=self._source_fps)
            else:
                state = adapter.process_frame(frame_bgr, frame_index, timestamp_sec)
            self.latest_module_states[module_key] = state
            for evt in state.get("events", []):
                evt = dict(evt)
                evt["module"] = module_key
                self.all_events.append(evt)

        if timestamp_sec - self._last_fusion_ts >= FUSION_UPDATE_INTERVAL_SEC:
            self.latest_fusion = self.fusion_engine.fuse_streaming_states(self.latest_module_states)
            self.latest_fusion = self._apply_fusion_smoothing(self.latest_fusion, timestamp_sec)
            self._last_fusion_ts = timestamp_sec
            if (
                not STORE_FULL_TIMELINE
                and frame_index % TIMELINE_SAMPLE_EVERY_N_FRAMES == 0
            ) or STORE_FULL_TIMELINE:
                self.score_timeline.append({
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "overall_score": self.latest_fusion["overall_score"],
                    "risk_level": self.latest_fusion["risk_level"],
                })
                for module_key, state in self.latest_module_states.items():
                  if state.get("status") == "READY" and state.get("result") is not None:
                    self.module_timeline[module_key].append({
                        "timestamp_sec": round(timestamp_sec, 2),
                        "frame_index": frame_index,
                        "risk_score": float(state["result"].risk_score),
                        "prediction": str(state["result"].prediction),
                        })

        if timestamp_sec >= self._next_segment_end:
            self._append_segment_summary(timestamp_sec)
            self._next_segment_end += SEGMENT_SUMMARY_SEC

        self._frames_processed = frame_index + 1
        self._peak_ram_mb = max(self._peak_ram_mb, _ram_mb())
        return self.get_latest_state()

    def _append_segment_summary(self, timestamp_sec: float) -> None:
        segment_start = max(0.0, timestamp_sec - SEGMENT_SUMMARY_SEC)
        scores = [entry["overall_score"] for entry in self.score_timeline if entry["timestamp_sec"] >= segment_start]
        summary = {
            "segment_start_sec": round(segment_start, 2),
            "segment_end_sec": round(timestamp_sec, 2),
            "latest_module_states": {
                key: {
                    "status": state.get("status"),
                    "prediction": None if not state.get("result") else state["result"].prediction,
                    "risk_score": None if not state.get("result") else state["result"].risk_score,
                }
                for key, state in self.latest_module_states.items()
            },
            "overall_score": None if not self.latest_fusion else self.latest_fusion["overall_score"],
            "risk_level": None if not self.latest_fusion else self.latest_fusion["risk_level"],
            "max_score_in_segment": max(scores) if scores else None,
            "events_in_segment": [e for e in self.all_events if segment_start <= e.get("time_sec", 0) <= timestamp_sec],
            "frames_processed": self._frames_processed,
            "processing_fps": self._effective_fps(),
        }
        if ENABLE_MEMORY_DIAGNOSTICS:
            summary["memory"] = {"ram_mb": _ram_mb(), "gpu_mb": _gpu_mem_mb()}
        self.segment_summaries.append(summary)
        if SAVE_SEGMENT_SUMMARIES:
            out_path = RECOVERY_OUTPUT_DIR / "segment_summaries.json"
            with open(out_path, "w") as f:
                json.dump(self.segment_summaries, f, indent=2)
            recovery = {
                "frame_index": self._frames_processed,
                "timestamp_sec": timestamp_sec,
                "latest_fusion": self.latest_fusion,
                "segment_count": len(self.segment_summaries),
            }
            with open(RECOVERY_OUTPUT_DIR / "recovery_state.json", "w") as f:
                json.dump(recovery, f, indent=2)

    def _effective_fps(self) -> float:
        elapsed = max(time.time() - self._wall_start, 1e-6)
        return self._frames_processed / elapsed


    def _compose_display_frame(self, frame_bgr: np.ndarray, timestamp_sec: float) -> np.ndarray:
        """Merge YOLO boxes (smoking/seatbelt) + classifier HUD. Orchestration only."""
        out = frame_bgr.copy()

        try:
            seatbelt_adapter = self.manager.get_adapter("seatbelt")
            for class_name, confidence, box_xyxy in getattr(seatbelt_adapter, "_stream_last_raw_detections", []):
                if class_name in getattr(seatbelt_adapter, "_stream_last_confirmed_classes", set()):
                    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(out, f"{class_name} {confidence:.2f}", (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception:
            pass

        try:
            smoking_adapter = self.manager.get_adapter("smoking")
            for class_name, confidence, box_xyxy in getattr(smoking_adapter, "_stream_last_raw", []):
                if class_name in getattr(smoking_adapter, "_stream_last_confirmed", set()):
                    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
                    color = (0, 0, 255) if class_name == "smoking" else (0, 140, 255)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{class_name} {confidence:.2f}", (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass

        hud_lines = []
        for module_key in self.MODULE_KEYS:
            state = self.latest_module_states.get(module_key, {})
            status = state.get("status", "N/A")
            result = state.get("result")
            if result is not None and not getattr(result, "error", None):
                hud_lines.append(f"{module_key}: {result.prediction} ({result.risk_score:.2f})")
            else:
                buf = state.get("buffer_len")
                extra = f" ({buf})" if buf is not None and status == "WARMING_UP" else ""
                hud_lines.append(f"{module_key}: {status}{extra}")

        y0 = 24
        for i, line in enumerate(hud_lines[:6]):
            cv2.putText(out, line, (10, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        score_text = "N/A"
        risk_text = "N/A"
        if self.latest_fusion:
            score_text = f"{self.latest_fusion['overall_score']:.1f}"
            risk_text = self.latest_fusion["risk_level"]
        cv2.putText(
            out,
            f"t={timestamp_sec:6.1f}s  Score={score_text}  Risk={risk_text}  FPS={self._effective_fps():.1f}",
            (10, out.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return out

    def _maybe_display(self, frame_bgr: np.ndarray, timestamp_sec: float) -> None:
        if not ENABLE_LIVE_DISPLAY:
            return
        min_interval = 1.0 / max(DISPLAY_MAX_FPS, 0.1)
        if timestamp_sec - self._last_display_ts < min_interval:
            return
        self._last_display_ts = timestamp_sec
        display_frame = self._compose_display_frame(frame_bgr, timestamp_sec)
        ok_enc, buf = cv2.imencode(".jpg", display_frame)
        if ok_enc:
            _ipy_clear_output(wait=True)
            _ipy_display(_IPyImage(data=buf.tobytes()))

    def run_recorded_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ) -> Dict[str, Any]:
        """Single-pass recorded-video sliding-window simulation (TA requirement)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.reset(fps=fps)
        self._wall_start = time.time()
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        progress = tqdm(total=total_frames or None, initial=start_frame, desc="Streaming inference", unit="frame")
        frame_index = start_frame
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_sec = frame_index / fps
                self.process_frame(frame, frame_index, timestamp_sec)
                self._maybe_display(frame, timestamp_sec)
                postfix = {}
                if self.latest_fusion:
                    postfix["Risk"] = self.latest_fusion["risk_level"].split()[0]
                    postfix["Score"] = f"{self.latest_fusion['overall_score']:.2f}"
                postfix["FPS"] = f"{self._effective_fps():.1f}"
                progress.set_postfix(postfix)
                progress.update(1)
                frame_index += 1
                if max_frames is not None and (frame_index - start_frame) >= max_frames:
                    break
        finally:
            cap.release()
            progress.close()
        return self.finalize()


    def run_webcam(self, source: int = 0, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Live camera sliding-window simulation (no ETA/percentage)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open webcam source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.reset(fps=fps)
        self._wall_start = time.time()
        progress = tqdm(desc="Live streaming", unit="frame")
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_sec = frame_index / fps
                self.process_frame(frame, frame_index, timestamp_sec)
                self._maybe_display(frame, timestamp_sec)
                progress.set_postfix(
                    Score=f"{self.latest_fusion['overall_score']:.1f}" if self.latest_fusion else "N/A",
                    FPS=f"{self._effective_fps():.1f}",
                )
                progress.update(1)
                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
        finally:
            cap.release()
            progress.close()
        return self.finalize()

    def get_latest_state(self) -> Dict[str, Any]:
        return {
            "module_states": self.latest_module_states,
            "fusion": self.latest_fusion,
            "frames_processed": self._frames_processed,
        }

    def finalize(self) -> Dict[str, Any]:
        if self.latest_fusion is None and self.latest_module_states:
            self.latest_fusion = self.fusion_engine.fuse_streaming_states(self.latest_module_states)

        prediction_results = []
        for module_key, state in self.latest_module_states.items():
            result = state.get("result")
            if result is None:
                # Module never produced a successful prediction this run (e.g. face
                # never detected, or buffer never filled for a very short clip) —
                # build a placeholder result instead of silently dropping it.
                adapter = self.manager.get_adapter(module_key)
                status = state.get("status", "UNAVAILABLE")
                reason = {
                    "WARMING_UP": "Insufficient frames buffered for this module",
                    "UNAVAILABLE": "Face not detected in the required window",
                }.get(status, "Module unavailable")
                result = PredictionResult(
                    module=adapter.MODULE_NAME,
                    prediction="Unavailable",
                    confidence=0.0,
                    risk_score=0.0,
                    metadata={"status": status, "reason": reason},
                    error=reason,
                )
            result.weight = RISK_FUSION_WEIGHTS.get(module_key, 0.0)
            result.metadata = dict(result.metadata or {})
            result.metadata["module_key"] = module_key
            prediction_results.append(result)

        return {
            "mode": "streaming",
            "prediction_results": prediction_results,
            "fusion_result": self.latest_fusion,
            "segment_summaries": self.segment_summaries,
            "score_timeline": list(self.score_timeline),
            "module_timeline": {k: list(v) for k, v in self.module_timeline.items()},
            "events": self.all_events,
            "frames_processed": self._frames_processed,
            "memory": {
                "initial_ram_mb": round(self._initial_ram_mb, 1),
                "peak_ram_mb": round(self._peak_ram_mb, 1),
                "final_ram_mb": round(_ram_mb(), 1),
                "gpu_mb": _gpu_mem_mb(),
            },
        }



print("StreamingWellnessOrchestrator ready.")

def print_session_summary(session_result: dict, session_type: str) -> None:
    """Integration/runtime summary — not model-accuracy reporting."""
    import time as _time
    elapsed = _time.time() - getattr(_orchestrator, "_wall_start", _time.time())
    fusion = session_result.get("fusion_result") or {}
    frames = session_result.get("frames_processed", 0)
    fps = frames / max(elapsed, 1e-6)
    module_errors = []
    for pr in session_result.get("prediction_results", []):
        if getattr(pr, "error", None):
            module_errors.append(f"{pr.module}: {pr.error}")
    active = fusion.get("modules_ready", fusion.get("module_status", {}))
    print("=" * 80)
    print("SESSION SUMMARY (integration runtime)")
    print("=" * 80)
    print(f"Session Type       : {session_type}")
    print(f"Frames Processed   : {frames}")
    print(f"Elapsed Time (s)   : {elapsed:.1f}")
    print(f"Average FPS        : {fps:.2f}")
    print(f"Modules Active     : {active}")
    print(f"Module Errors      : {module_errors or 'none'}")
    print(f"Final Fusion Score : {fusion.get('overall_score', 'N/A')}")
    print(f"Risk Level         : {fusion.get('risk_level', 'N/A')}")
    if fusion.get("r_total") is not None:
        print(f"R_total (Σ R_i)    : {fusion.get('r_total')}")
        print(f"Fusion k           : {fusion.get('fusion_k', FUSION_EXPONENTIAL_K)}")
    mem = session_result.get("memory", {})
    if mem:
        print(f"Peak RAM (MB)      : {mem.get('peak_ram_mb', 'N/A')}")
    print("=" * 80)


# ============================================================
# GLUE — what app.py imports and calls
# ============================================================
# Suppress noisy per-frame logging in the web app
logging.getLogger("DriverWellnessAI").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Shared singletons (built once, reused across requests)
risk_fusion_engine = DriverWellnessRiskFusion()
module_manager = DriverWellnessModuleManager()
_orchestrator: Optional["StreamingWellnessOrchestrator"] = None

# Live-webcam streaming state (Gradio drives one frame at a time)
_live_frame_index: int = 0
_live_active: bool = False


def _register_all(manager: DriverWellnessModuleManager) -> None:
    """Register the 5 adapters onto a manager (does NOT load weights yet)."""
    manager.register_adapter(
        "video_fatigue", VideoFatigueAdapter(MODEL_REGISTRY["video_fatigue"]["checkpoint"], device=DEVICE))
    manager.register_adapter(
        "landmark_fatigue", LandmarkFatigueAdapter(MODEL_REGISTRY["landmark_fatigue"]["checkpoint"], device=DEVICE))
    manager.register_adapter(
        "driver_activity", DriverActivityAdapter(MODEL_REGISTRY["driver_activity"]["checkpoint"], device=DEVICE))
    manager.register_adapter(
        "smoking", SmokingAdapter(MODEL_REGISTRY["smoking"]["checkpoint"], device=DEVICE))
    manager.register_adapter(
        "seatbelt", SeatBeltPhoneDetectionAdapter(MODEL_REGISTRY["seatbelt"]["checkpoint"], device=DEVICE))


def build_manager() -> DriverWellnessModuleManager:
    """Register + warm up all 5 models. Call ONCE at app startup."""
    global _orchestrator
    _register_all(module_manager)
    module_manager.load_all()  # triggers warmup() -> loads every checkpoint
    _orchestrator = StreamingWellnessOrchestrator(module_manager, risk_fusion_engine)
    return module_manager


def _resolve_ffmpeg_exe() -> Optional[str]:
    """Prefer a system ffmpeg on PATH; fall back to imageio-ffmpeg's bundled static binary."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_for_browser(raw_path: str, fps: float) -> str:
    """
    Re-encode a raw OpenCV-written .mp4 (codec 'mp4v'/MPEG-4 Part 2) into
    H.264/yuv420p, which is what browsers actually support for inline
    <video> playback. Without this step, Gradio's preview player shows a
    blank/broken video even though the file is valid and plays fine in a
    desktop media player after downloading -- which is exactly the codec
    mismatch, not a Gradio bug.

    Falls back to the raw file untouched if no ffmpeg binary is available or
    the transcode fails for any reason, so recorded-video analysis never breaks.
    """
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if not ffmpeg_exe:
        logger.warning("ffmpeg not available; serving raw mp4v output (may not preview inline).")
        return raw_path

    final_path = str(Path(tempfile.gettempdir()) / f"{Path(raw_path).stem}_h264.mp4")
    cmd = [
        ffmpeg_exe, "-y",
        "-i", raw_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        final_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    except Exception as exc:
        logger.warning("ffmpeg transcode to H.264 failed (%s); serving raw mp4v output.", exc)
        return raw_path

    try:
        os.remove(raw_path)
    except OSError:
        pass
    return final_path


def run_recorded_video(manager: DriverWellnessModuleManager, video_path: str):
    """
    Stream the uploaded video through all modules, write an annotated .mp4,
    and return (annotated_path, summary_dict).

    Reuses the orchestrator's per-frame processing and HUD compositor; instead
    of IPython inline display, annotated frames are written to a VideoWriter.
    The raw writer output is then transcoded to browser-playable H.264 so the
    Gradio preview player can show it inline (not just after downloading).
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = StreamingWellnessOrchestrator(manager, risk_fusion_engine)
    orch = _orchestrator

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoDecodeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    raw_path = str(Path(tempfile.gettempdir()) / f"annotated_{int(time.time())}_raw.mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    orch.reset(fps=fps)
    orch._wall_start = time.time()
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_sec = frame_index / fps
            orch.process_frame(frame, frame_index, timestamp_sec)
            annotated = orch._compose_display_frame(frame, timestamp_sec)
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    out_path = _transcode_for_browser(raw_path, fps)
    result = orch.finalize()
    return out_path, _fusion_summary_dict(result, frames_fallback=frame_index)


# ------------------------------------------------------------
# Shared summary formatting (recorded + live) — Option A fields
# ------------------------------------------------------------
def _fusion_summary_dict(result: Dict[str, Any], frames_fallback: int = 0) -> Dict[str, Any]:
    fusion = result.get("fusion_result") or {}
    summary: Dict[str, Any] = {
        "Overall Score": f"{fusion.get('overall_score', 0):.1f} / 100",
        "Risk Level": fusion.get("risk_level", "N/A"),
        "Frames processed": result.get("frames_processed", frames_fallback),
        "Modules ready": ", ".join(fusion.get("modules_ready", [])) or "none",
        "Modules warming up": ", ".join(fusion.get("modules_warming_up", [])) or "none",
    }
    if fusion.get("r_total") is not None:
        summary["R_total (sum of Ri)"] = fusion.get("r_total")
        summary["Fusion k"] = fusion.get("fusion_k", FUSION_EXPONENTIAL_K)
    # Per-module breakdown (new Option A fields)
    for contrib in fusion.get("contributions", []):
        name = contrib.get("module", "?")
        summary[f"  • {name}"] = (
            f"{contrib.get('prediction')} "
            f"(event_risk {contrib.get('event_risk', 0):.2f}, "
            f"severity {contrib.get('severity_weight', 0):.0f})"
        )
    return summary


# ============================================================
# LIVE WEBCAM STREAMING — driven one frame at a time by Gradio
# ============================================================
# The notebook's live mode used Colab's browser-JS webcam or a local
# cv2.VideoCapture(0) loop. Neither is available inside a Hugging Face
# Space, so here the browser streams frames to us through Gradio and we
# reuse the exact same orchestrator.process_frame + _compose_display_frame
# pipeline per frame.
LIVE_STREAM_FPS = LIVE_WEBCAM_FPS  # timestamp assumption for the fusion timeline


def start_live_session(fps: Optional[float] = None) -> None:
    """(Re)initialise the shared orchestrator for a fresh live webcam session."""
    global _orchestrator, _live_frame_index, _live_active
    if _orchestrator is None:
        _orchestrator = StreamingWellnessOrchestrator(module_manager, risk_fusion_engine)
    fps = float(fps or LIVE_STREAM_FPS)
    _orchestrator.reset(fps=fps)
    _orchestrator._wall_start = time.time()
    _live_frame_index = 0
    _live_active = True


def process_live_frame(frame_rgb, fps: Optional[float] = None):
    """
    Consume ONE webcam frame (RGB HxWx3, as delivered by Gradio) and return
    (annotated_rgb, summary_text).

    Auto-starts a session on the first frame, so it is safe to wire directly
    to a streaming ``gr.Image``. Returns (None, message) when no frame is
    supplied (e.g. the stream just stopped).
    """
    global _orchestrator, _live_frame_index, _live_active
    if frame_rgb is None:
        return None, "Waiting for webcam frames..."
    if _orchestrator is None or not _live_active:
        start_live_session(fps=fps)

    fps = float(fps or _orchestrator._source_fps or LIVE_STREAM_FPS)
    # Gradio hands us RGB; the pipeline (like cv2/YOLO/MediaPipe) expects BGR.
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    timestamp_sec = _live_frame_index / fps
    _orchestrator.process_frame(frame_bgr, _live_frame_index, timestamp_sec)
    annotated_bgr = _orchestrator._compose_display_frame(frame_bgr, timestamp_sec)
    _live_frame_index += 1

    fusion = _orchestrator.latest_fusion or {}
    lines = [
        f"Overall Score : {fusion.get('overall_score', 0):.1f} / 100",
        f"Risk Level    : {fusion.get('risk_level', 'N/A')}",
        f"Frames        : {_live_frame_index}",
        f"FPS (proc)    : {_orchestrator._effective_fps():.1f}",
    ]
    warming = fusion.get("modules_warming_up", [])
    if warming:
        lines.append("Warming up    : " + ", ".join(warming))
    for contrib in fusion.get("contributions", []):
        lines.append(
            f"  • {contrib.get('module')}: {contrib.get('prediction')} "
            f"(risk {contrib.get('event_risk', 0):.2f})"
        )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, "\n".join(lines)


def stop_live_session() -> str:
    """Finalise the current live session and return a formatted summary string."""
    global _orchestrator, _live_active
    if _orchestrator is None or not _live_active:
        return "No active live session."
    _live_active = False
    result = _orchestrator.finalize()
    summary = _fusion_summary_dict(result, frames_fallback=_live_frame_index)
    return "\n".join(f"{k}: {v}" for k, v in summary.items())
