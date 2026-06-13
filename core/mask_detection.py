import cv2
import numpy as np
import threading

from uniface.parsing import BiSeNet
from uniface.constants import ParsingWeights

from core.camera import backproject, Camera, coordinate_transform, load_camera


_PARSER_LOCK = threading.Lock()
_PARSER_CACHE = None

FACE_LABELS      = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13]
FACE_NECK_LABELS = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14]


def get_parser():
    """全局缓存 BiSeNet 实例, 避免重复加载权重."""
    global _PARSER_CACHE
    with _PARSER_LOCK:
        if _PARSER_CACHE is None:
            _PARSER_CACHE = BiSeNet(model_name=ParsingWeights.RESNET34)
    return _PARSER_CACHE


def _mask_bbox(parsing, labels, pad=20):
    """从 parsing 标签区域计算 xyxy bbox; 无有效像素时返回 None."""
    mask = np.isin(parsing, labels)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None

    h, w = parsing.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w - 1, int(xs.max()) + pad)
    y2 = min(h - 1, int(ys.max()) + pad)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def _backproject_mask(parsing, depth, camera, labels):
    mask = np.isin(parsing, labels)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    uvs = np.stack([xs, ys], axis=1).astype(np.float32)
    pts_cam, _ = backproject(uvs, depth, camera)
    return coordinate_transform(pts_cam)


def _face_neck_crop_xyxy(rgb, bbox_xywh):
    """YuNet bbox: extend downward to 2x height, then expand the resulting box by 1.2."""
    h_img, w_img = rgb.shape[:2]
    x, y, w, h = [float(v) for v in bbox_xywh]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + 2.0 * h

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    crop_w = (x2 - x1) * 1.2
    crop_h = (y2 - y1) * 1.2

    x1 = max(0, int(round(cx - crop_w / 2.0)))
    y1 = max(0, int(round(cy - crop_h / 2.0)))
    x2 = min(w_img, int(round(cx + crop_w / 2.0)))
    y2 = min(h_img, int(round(cy + crop_h / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_crop_mapped(parser, rgb, bbox_xywh):
    crop_xyxy = _face_neck_crop_xyxy(rgb, bbox_xywh)
    if crop_xyxy is None:
        return parser.parse(rgb)

    x1, y1, x2, y2 = crop_xyxy
    parsing_crop = parser.parse(rgb[y1:y2, x1:x2])
    parsing = np.zeros(rgb.shape[:2], dtype=parsing_crop.dtype)
    parsing[y1:y2, x1:x2] = parsing_crop
    return parsing


def get_face_data_single(camera, rgb, depth, bbox_xywh=None):
    """
    单帧人脸分割结果：
      - bbox_face_xyxy: 由 FACE_LABELS 计算的 2D bbox (xyxy)
      - pts_face: face 点云 (不含颈部)
      - pts_face_neck: face+neck 点云
      - valid: 是否识别到可用面部区域（bbox 非空且 face 点云非空）
    """
    parser = get_parser()
    parsing = _parse_crop_mapped(parser, rgb, bbox_xywh) if bbox_xywh is not None else parser.parse(rgb)

    bbox_face = _mask_bbox(parsing, FACE_LABELS)
    pts_face = _backproject_mask(parsing, depth, camera, FACE_LABELS)
    pts_face_neck = _backproject_mask(parsing, depth, camera, FACE_NECK_LABELS)

    valid = bbox_face is not None and pts_face.shape[0] > 0
    return {
        "bbox_face_xyxy": bbox_face,
        "pts_face": pts_face,
        "pts_face_neck": pts_face_neck,
        "valid": bool(valid),
    }


def get_face_mask(
    camera,
    rgb_closed,
    rgb_open,
    depth_closed,
    depth_open,
    bbox_closed_xywh=None,
    bbox_open_xywh=None,
):
    """
    返回两种点云：
      face (不含脖子) — 用于 shape/expr 优化
      face_neck (含脖子) — 用于 pose 优化
    """

    closed = get_face_data_single(camera, rgb_closed, depth_closed, bbox_closed_xywh)
    opened = get_face_data_single(camera, rgb_open, depth_open, bbox_open_xywh)
    return (
        closed["pts_face"],
        opened["pts_face"],
        closed["pts_face_neck"],
        opened["pts_face_neck"],
    )
