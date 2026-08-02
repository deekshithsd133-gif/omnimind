"""Unified object-detection engine: three nano YOLO models run per frame and
their outputs are normalized into a single `Detection` list.

Model provenance (see C:\\Users\\Deekshit\\omni-mind-project\\README.md for full detail):
  - yolo11n.pt              Ultralytics COCO-pretrained. Real classes used: person, knife, baseball bat.
  - ppe_helmet_mask.pt       Tanishjain9/yolov8n-ppe-detection-6classes (MIT, HuggingFace). Real: helmet, mask.
  - threat_gun_knife.pt      Subh775/Threat-Detection-YOLOv8n (MIT, HuggingFace). Real: gun, knife, explosive, grenade.

No open, pip-installable model exists for "iron rod" specifically — that
signal comes from `models.blunt_object_heuristic`, which is intentionally
capped below the weapon confidence threshold and can never alone trigger a
lockdown. See that module's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from config.settings import settings
from utils.logger import get_logger

log = get_logger("yolo_engine")

# category -> (min confidence from config, is this a "weapon" for Step 6 lockdown purposes)
CATEGORY_META = {
    "person": (settings.conf_person, False),
    "helmet": (settings.conf_helmet, False),
    "mask": (settings.conf_mask, False),
    "weapon_gun": (settings.conf_weapon, True),
    "weapon_knife": (settings.conf_weapon, True),
    "weapon_baseball_bat": (settings.conf_weapon, True),
    "weapon_explosive": (settings.conf_weapon, True),
    "weapon_grenade": (settings.conf_weapon, True),
}

_COCO_MAP = {"person": "person", "knife": "weapon_knife", "baseball bat": "weapon_baseball_bat"}
_PPE_MAP = {"helmet": "helmet", "mask": "mask"}
_THREAT_MAP = {"gun": "weapon_gun", "knife": "weapon_knife", "explosion": "weapon_explosive", "grenade": "weapon_grenade"}


@dataclass
class Detection:
    category: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    source_model: str
    heuristic: bool = False

    @property
    def is_weapon(self) -> bool:
        return CATEGORY_META.get(self.category, (0.0, False))[1]

    @property
    def passes_threshold(self) -> bool:
        min_conf = CATEGORY_META.get(self.category, (0.99, False))[0]
        return self.confidence >= min_conf

    @property
    def worth_showing(self) -> bool:
        """Half the real alarm threshold — below this a detection is just
        pre-filter noise (yolo_engine runs YOLO at conf=0.25 internally) and
        drawing it on the dashboard as e.g. "weapon 27%" is actively
        misleading. Above this but below passes_threshold, the UI shows it
        as an unconfirmed/dashed box; only passes_threshold detections that
        also clear the rolling-window confirmation get a solid alarm box."""
        min_conf = CATEGORY_META.get(self.category, (0.99, False))[0]
        return self.confidence >= min_conf * 0.5


def _select_device() -> str:
    if settings.device_preference == "cpu":
        return "cpu"
    if settings.device_preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        log.warning("CUDA requested but not available — falling back to CPU")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class DetectionEngine:
    """Loads all YOLO models once and runs them per frame. Not thread-safe for
    concurrent `.infer()` calls from multiple threads — the detection pipeline
    owns a single dedicated inference thread by design (see workflow/pipeline.py)."""

    def __init__(self) -> None:
        self.device = _select_device()
        log.info("DetectionEngine initializing on device=%s", self.device)

        self.coco_model = YOLO(str(settings.weights_dir / "yolo11n.pt"))
        self.ppe_model = YOLO(str(settings.weights_dir / "ppe_helmet_mask.pt"))
        self.threat_model = YOLO(str(settings.weights_dir / "threat_gun_knife.pt"))

        for model in (self.coco_model, self.ppe_model, self.threat_model):
            try:
                model.to(self.device)
            except Exception:
                log.exception("Failed to move a model to %s; it will run on its default device", self.device)

        log.info("DetectionEngine ready")

    def _run_one(self, model: YOLO, frame: np.ndarray, name_map: dict[str, str], source_label: str) -> list[Detection]:
        out: list[Detection] = []
        try:
            results = model.predict(
                frame,
                imgsz=settings.inference_img_size,
                verbose=False,
                device=self.device,
                conf=0.25,  # low pre-filter; real gating happens via CATEGORY_META thresholds downstream
                quantize=16 if self.device == "cuda" else None,  # fp16: ~free speedup on a CUDA GPU, no meaningful accuracy loss for these nano models (this ultralytics version's replacement for the deprecated `half=` arg)
                max_det=20,  # single-camera ATM booth scene never legitimately has more objects than this; caps NMS/postprocessing cost
            )
        except Exception:
            log.exception("Inference failed for %s", source_label)
            return out

        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls.item())
                raw_name = names.get(cls_idx, str(cls_idx)).lower()
                category = name_map.get(raw_name)
                if category is None:
                    continue
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(Detection(category=category, confidence=conf, bbox=(x1, y1, x2, y2), source_model=source_label))
        return out

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Runs all three models on one BGR frame and returns merged detections."""
        detections: list[Detection] = []
        coco_dets = self._run_one(self.coco_model, frame, _COCO_MAP, "yolo11n-coco")
        detections += coco_dets
        # Helmet/mask detections are only ever meaningful once associated
        # with a person (workflow/pipeline.py matches them by bbox
        # containment) — skip the PPE model entirely on frames where the
        # COCO pass found no person-shaped object at all, which is the
        # common case for an otherwise-empty ATM booth. This keeps average
        # per-frame GPU load down so the pipeline doesn't fall behind and
        # stays responsive the moment someone actually walks into frame.
        # The threat/weapon model always runs regardless of this check —
        # weapon detection is the highest-priority security signal (see
        # "Step 6: weapon detection takes priority over everything" in
        # workflow/pipeline.py) and must never be skipped speculatively.
        if any(d.category == "person" for d in coco_dets):
            detections += self._run_one(self.ppe_model, frame, _PPE_MAP, "ppe-yolov8n")
        detections += self._run_one(self.threat_model, frame, _THREAT_MAP, "threat-yolov8n")
        return detections
