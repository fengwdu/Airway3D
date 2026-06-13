import numpy as np
import numpy as np

from .camera import Camera, backproject_window, coordinate_transform, load_camera
from third_party.tufa import Test

# 本模块在主流程中的职责：
# 通过 TUFA 预测 68 点 2D landmark，并结合深度图反投影为 3D landmark，
# 作为 mesh 拟合的核心几何约束。


def infer_landmark_rgbd(
    rgb_closed,
    rgb_open,
    depth_closed,
    depth_open,
    camera,
    bbox_closed=None,
    bbox_open=None,
):
    """
    输出:
    - TUFA 2D 关键点
    - 深度反投影 3D
    """

    pred_closed, pred_open = Test.get_lmk(
        rgb_closed, rgb_open,
        bbox_closed=bbox_closed, bbox_open=bbox_open,
    )

    lmk2d_closed = np.asarray(pred_closed, dtype=np.float32)
    lmk2d_open = np.asarray(pred_open, dtype=np.float32)

    lmk3d_closed, mask_closed = backproject_window(
        lmk2d_closed, depth_closed, camera, window=2, trim=0, keep_shape=True
    )
    lmk3d_open, mask_open = backproject_window(
        lmk2d_open, depth_open, camera, window=2, trim=0, keep_shape=True
    )

    lmk3d_closed = coordinate_transform(lmk3d_closed)
    lmk3d_open = coordinate_transform(lmk3d_open)

    return lmk2d_closed, lmk2d_open, lmk3d_closed, lmk3d_open, mask_closed, mask_open


def validate_landmark_single_frame(rgb, depth, camera, bbox_xyxy):
    """
    单帧关键点校验：
      - 使用给定 bbox 检测 68 点
      - 要求 17-47 中可用于 Umeyama 的点不少于 8 个
      - 要求后续指标/约束所需语义点有效
    """
    if bbox_xyxy is None:
        return False

    try:
        pred, _ = Test.get_lmk(rgb, rgb, bbox_closed=bbox_xyxy, bbox_open=bbox_xyxy)
    except Exception:
        return False

    lmk2d = np.asarray(pred, dtype=np.float32)
    _, mask = backproject_window(lmk2d, depth, camera, window=2, trim=0, keep_shape=True)
    if mask.size != 68:
        return False
    align_mask = mask[17:48].copy()
    align_mask[[0, 1, 2, 7, 8, 9]] = False
    required = np.array([8, 21, 22, 33, 50, 51, 52, 56, 57, 58], dtype=np.int64)
    return bool(align_mask.sum() >= 8 and mask[required].all())
