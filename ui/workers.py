import os
import traceback
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal as Signal


class CameraThread(QThread):
    """Background thread that continuously reads camera frames and emits them via signal."""

    frame_ready = Signal(np.ndarray, np.ndarray, object)  # rgb, depth, camera

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False

    def run(self):
        from core.capture import OrbbecCapture

        # 相机线程职责：
        # 持续读 Orbbec 帧并把 RGB/Depth 发给采集页预览。
        cap = OrbbecCapture(timeout_ms=100)
        cap.start()
        self._running = True

        while self._running:
            out = cap.read()
            if out is None:
                continue
            camera, rgb, depth = out
            self.frame_ready.emit(rgb.copy(), depth.copy(), camera)

        cap.stop()

    def stop(self):
        self._running = False
        self.wait()


class PipelineWorker(QThread):
    """Background thread that runs the full pipeline: landmarks -> point cloud -> fitting -> metrics -> render."""

    progress = Signal(str)                              # 当前步骤描述
    finished = Signal(dict, object, object, object)    # metrics, proj_closed, proj_open, aruco_corners
    error    = Signal(str)                              # 错误 traceback

    def __init__(self, rgb_closed, rgb_open, depth_closed, depth_open, camera,
                 bbox_closed_xywh=None, bbox_open_xywh=None, landmarks=None, parent=None):
        super().__init__(parent)
        self.rgb_closed       = rgb_closed
        self.rgb_open         = rgb_open
        self.depth_closed     = depth_closed
        self.depth_open       = depth_open
        self.camera           = camera
        self.bbox_closed_xywh = bbox_closed_xywh
        self.bbox_open_xywh   = bbox_open_xywh
        self.landmarks        = landmarks

    @staticmethod
    def _xywh_to_xyxy(bbox):
        """[x, y, w, h] → [x1, y1, x2, y2]，供 infer_landmark_rgbd 使用。"""
        if bbox is None:
            return None
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h], dtype=np.int32)

    def run(self):
        try:
            import os
            from core.mask_detection import get_face_data_single
            from core.aruco        import detect_aruco
            from core.fitting      import train
            from core.render       import auto_render_mesh
            from core.metrics      import metrics_calculate

            # Step 1: 复用采集阶段已通过质检的 TUFA 关键点与 3D 反投影结果。
            self.progress.emit("正在读取关键点...")
            if self.landmarks is None:
                raise RuntimeError("未获取到关键点结果，请重新采集")

            lmk2d_c = self.landmarks["lmk2d_closed"]
            lmk2d_o = self.landmarks["lmk2d_open"]
            lmk3d_c = self.landmarks["lmk3d_closed"]
            lmk3d_o = self.landmarks["lmk3d_open"]
            mask_c = self.landmarks["mask_closed"]
            mask_o = self.landmarks["mask_open"]

            # Step 2: ArUco 检测喉结锚点，后续指标计算优先使用该 3D 点。
            self.progress.emit("正在检测 ArUco 标记...")
            corners_c, _, adam_3d = detect_aruco(self.rgb_closed, self.depth_closed, self.camera)
            aruco_corners = {"closed": corners_c, "open": None}

            if adam_3d is None:
                raise RuntimeError("未检测到 ArUco 标记或喉结点反投影失败，请重新采集闭口图像")
            print(f"[ArUco] 喉结 3D 坐标 (模型): {adam_3d}")

            # Step 3: 人脸区域分割，得到面部点云/颈部点云。
            self.progress.emit("正在分割面部区域...")
            face_c = get_face_data_single(
                self.camera, self.rgb_closed, self.depth_closed,
                bbox_xywh=self.bbox_closed_xywh,
            )
            face_o = get_face_data_single(
                self.camera, self.rgb_open, self.depth_open,
                bbox_xywh=self.bbox_open_xywh,
            )
            if (not face_c["valid"]) or (not face_o["valid"]):
                raise RuntimeError("未识别到人脸，请重新采集")

            # Step 4: 整理拟合所需点云（face 与 face+neck）。
            self.progress.emit("正在整理点云...")
            pts_face_c = face_c["pts_face"]
            pts_face_o = face_o["pts_face"]
            pts_neck_c = face_c["pts_face_neck"]
            pts_neck_o = face_o["pts_face_neck"]

            # Step 5: FLAME 拟合（零初始化后做多阶段优化）。
            self.progress.emit("正在拟合模型...")
            mesh_closed, mesh_open = train(
                lmk3d_c, lmk3d_o,
                lmk2d_c, lmk2d_o,
                mask_c, mask_o,
                pts_neck_c, pts_neck_o,
                camera=self.camera,
                rgb_closed=self.rgb_closed,
                rgb_open=self.rgb_open,
                ptcloud_face_closed=pts_face_c,
                ptcloud_face_open=pts_face_o,
            )

            os.makedirs("./output/closed", exist_ok=True)
            os.makedirs("./output/open", exist_ok=True)
            mesh_closed.export("./output/closed/mesh_closed.obj")
            mesh_open.export("./output/open/mesh_open.obj")

            # Step 6: 基于闭口/张口网格计算临床相关指标。
            self.progress.emit("正在计算指标...")
            (mouth_opening, thyromental, head_tilt_angle,
             jaw_activity_angle, jaw_activity_distance,
             neck_circumference, tongue_thickness, length) = metrics_calculate(
                mesh_closed, mesh_open, adam_3d, lmk3d_c, lmk3d_o
            )

            metrics = {
                "张口度":     f"{mouth_opening:.2f} cm",
                "甲颌距离":   f"{thyromental:.2f} cm",
                "头后仰角度": f"{head_tilt_angle:.2f}°",
                "下颌活动角度": f"{jaw_activity_angle:.2f}°",
                "下颌活动度": f"{jaw_activity_distance:.2f} cm",
                "颈围":       f"{neck_circumference:.2f} cm",
                "舌厚度":     f"{tongue_thickness:.2f} cm",
                "声门视野":   f"{length:.2f} cm",
            }

            # Step 7: 离屏渲染网格投影图，用于结果页展示。
            self.progress.emit("正在渲染结果...")
            proj_closed = auto_render_mesh(mesh_closed.vertices, mesh_closed.faces, 800, 800)
            proj_open   = auto_render_mesh(mesh_open.vertices,   mesh_open.faces,   800, 800)

            self.finished.emit(metrics, proj_closed, proj_open, aruco_corners)

        except Exception:
            self.error.emit(traceback.format_exc())
