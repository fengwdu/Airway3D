import cv2
import numpy as np

from .camera import backproject_bilinear, coordinate_transform

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def detect_aruco(rgb, depth, camera):
    """
    检测 DICT_4X4_50 ArUco 码并反投影中心点到 3D 模型坐标.

    返回:
        corners_2d: (4, 2) 四个角点像素坐标, 未检测到返回 None
        center_2d:  (2,) 中心点像素坐标, 未检测到返回 None
        center_3d:  (3,) 模型坐标系下的 3D 坐标, 反投影失败返回 None
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None, None, None

    corners_2d = corners_list[0][0]  # (4, 2) 取第一个检测到的码
    center_2d = corners_2d.mean(axis=0)  # (2,)

    center_uv = center_2d.reshape(1, 2).astype(np.float32)
    pts_3d, valid = backproject_bilinear(center_uv, depth, camera)

    if valid.sum() == 0:
        return corners_2d, center_2d, None

    center_3d = coordinate_transform(pts_3d)[0]  # (3,)

    return corners_2d, center_2d, center_3d


def draw_aruco_on_rgb(rgb, corners_2d):
    """
    在 RGB 图像上绘制 ArUco 检测框, 返回带标注的副本.
    corners_2d: (4, 2) 或 None
    """
    vis = rgb.copy()
    if corners_2d is None:
        return vis

    pts = corners_2d.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    center = corners_2d.mean(axis=0).astype(np.int32)
    cv2.circle(vis, (center[0], center[1]), 4, (0, 0, 255), -1)

    return vis
