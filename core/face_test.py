from core.config import bbox_model_path, score_threshold, nms_threshold, top_k
import cv2
import numpy as np
import threading


_DETECTOR = None
_DETECTOR_SIZE = None
_DETECTOR_LOCK = threading.Lock()


def _get_detector(size):
    """缓存 YuNet detector，实时预览时避免每帧重复加载 ONNX。"""
    global _DETECTOR, _DETECTOR_SIZE
    if _DETECTOR is None:
        _DETECTOR = cv2.FaceDetectorYN.create(
            str(bbox_model_path),
            "",
            size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        _DETECTOR_SIZE = size
    elif _DETECTOR_SIZE != size:
        _DETECTOR.setInputSize(size)
        _DETECTOR_SIZE = size
    return _DETECTOR


def detect_face_bbox(
    cv_bgr: np.ndarray,
):
    h, w = cv_bgr.shape[:2]
    with _DETECTOR_LOCK:
        detector = _get_detector((w, h))
        _, faces = detector.detect(cv_bgr)


    if faces is None:
        return None

    face = max(faces, key=lambda f: f[2] * f[3])

    return face[:4].astype(int)            # x, y, w, h
