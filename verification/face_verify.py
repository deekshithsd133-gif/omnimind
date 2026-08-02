"""Face verification via FaceNet (InceptionResnetV1, VGGFace2-pretrained)
embeddings and cosine similarity — the ArcFace/FaceNet family the spec calls
for. Face *detection*/cropping is done upstream by MediaPipe (liveness
module) so this module only ever sees an already-localized face crop.

Only 512-d embedding vectors are ever persisted (Fernet-encrypted, see
utils.security). No raw face image is written to disk or database.
"""
from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from config.settings import settings
from utils.logger import get_logger

log = get_logger("face_verify")


def _standardize(face_bgr: np.ndarray) -> torch.Tensor:
    import cv2

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (160, 160))
    tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0  # standard FaceNet "prewhiten"
    return tensor.unsqueeze(0)


class FaceVerifier:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        log.info("FaceVerifier ready on device=%s", device)

    @torch.no_grad()
    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        tensor = _standardize(face_bgr).to(self.device)
        embedding = self.model(tensor)
        vec = embedding.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def matches(self, probe: np.ndarray, enrolled: np.ndarray) -> tuple[bool, float]:
        sim = self.cosine_similarity(probe, enrolled)
        return sim >= settings.face_match_cosine_threshold, sim
