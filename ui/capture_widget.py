import cv2
import numpy as np
import threading
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QMessageBox, QGridLayout,
)
from PyQt6.QtCore import pyqtSignal as Signal, Qt, QThread
from PyQt6.QtGui import QKeySequence, QShortcut

from .utils import ImageLabel, depth_to_rgb
from core.face_test import detect_face_bbox
from core.aruco import detect_aruco
from core.camera import backproject_window, coordinate_transform, merge_depth_frames
from third_party.tufa.Test import get_lmk_single

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)

_BTN_STYLE = """
    QPushButton {
        background: #1976d2; color: white;
        border-radius: 6px; font-size: 14px;
        padding: 8px;
    }
    QPushButton:hover   { background: #1565c0; }
    QPushButton:pressed { background: #0d47a1; }
    QPushButton:disabled { background: #e0e0e0; color: #9e9e9e; }
"""

_ALIGN_LMK_COUNT = 31
_ALIGN_EXCLUDED_LMK_68 = np.array([17, 18, 19, 24, 25, 26], dtype=np.int64)
_MIN_ALIGN_LANDMARKS = 8
_REQUIRED_LMK_68 = np.array([8, 21, 22, 33, 50, 51, 52, 56, 57, 58], dtype=np.int64)


class FaceBBoxThread(QThread):
    """后台跑 YuNet，UI 线程只消费最近一次 bbox。"""

    bbox_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._frame = None
        self._lock = threading.Lock()

    def submit(self, rgb: np.ndarray):
        with self._lock:
            self._frame = rgb.copy()

    def run(self):
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None

            if frame is None:
                self.msleep(30)
                continue

            try:
                bbox = detect_face_bbox(frame)
            except Exception:
                bbox = None
            self.bbox_ready.emit(bbox)

    def stop(self):
        self._running = False
        self.wait()


class CaptureWidget(QWidget):
    """
    Page 1: live camera preview + buttons to capture closed/open frames.

    Signal capture_done(rgb_closed, rgb_open, depth_closed, depth_open, camera,
                        bbox_closed, bbox_open, landmarks)
    """

    # rgb_c, rgb_o, dep_c, dep_o, camera, bbox_c_xywh, bbox_o_xywh, landmarks
    capture_done = Signal(object, object, object, object, object, object, object, object)

    _ARUCO_HOLD_FRAMES = 10      # 检测丢失后保持绿框的检测次数
    _BBOX_PREVIEW_INTERVAL = 10  # 预览阶段每 N 帧跑一次 YuNet
    _ARUCO_PREVIEW_INTERVAL = 6  # 预览阶段每 N 帧检测一次 ArUco
    _DEPTH_PREVIEW_INTERVAL = 3  # depth 伪彩色每 N 帧刷新一次

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stage = 0          # 0: waiting for closed, 1: waiting for open
        self._camera = None
        self._rgb = None
        self._depth = None
        self._rgb_closed = None
        self._depth_closed = None
        self._bbox_closed_xywh = None
        self._lmk2d_closed = None
        self._lmk3d_closed = None
        self._lmk_mask_closed = None
        self._aruco_cache = None   # 上次检测到的 corners/ids
        self._aruco_age = 0        # 距上次检测成功的帧数
        self._frame_buffer = []
        self._preview_bbox = None
        self._frame_count = 0
        self._warning_active = False
        self._build_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._space_shortcut = QShortcut(QKeySequence("Space"), self)
        self._space_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._space_shortcut.activated.connect(self._on_capture)
        self._bbox_thread = FaceBBoxThread(self)
        self._bbox_thread.bbox_ready.connect(self._on_bbox_ready)
        self._bbox_thread.start()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # left: preview area (closed row + open row)
        preview = QGridLayout()
        preview.setSpacing(8)
        self.lbl_closed_rgb   = ImageLabel("闭口 RGB")
        self.lbl_closed_depth = ImageLabel("闭口 Depth")
        self.lbl_open_rgb     = ImageLabel("张口 RGB")
        self.lbl_open_depth   = ImageLabel("张口 Depth")
        preview.addWidget(self.lbl_closed_rgb,   0, 0)
        preview.addWidget(self.lbl_closed_depth, 0, 1)
        preview.addWidget(self.lbl_open_rgb,     1, 0)
        preview.addWidget(self.lbl_open_depth,   1, 1)
        preview.setRowStretch(0, 1)
        preview.setRowStretch(1, 1)
        preview.setColumnStretch(0, 1)
        preview.setColumnStretch(1, 1)
        root.addLayout(preview, stretch=3)

        # right: status sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background: #ffffff; border: none; border-radius: 8px;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(20, 30, 20, 30)
        sb.setSpacing(12)

        self.lbl_title = QLabel("采集闭口图像")
        self.lbl_title.setStyleSheet(
            "color: #212121; font-size: 16px; font-weight: bold;"
        )
        self.lbl_title.setWordWrap(True)

        lbl_hint = QLabel("点击按钮或按空格键")
        lbl_hint.setStyleSheet("color: #757575; font-size: 12px;")

        self.btn_capture = QPushButton("采集闭口")
        self.btn_capture.setFixedHeight(44)
        self.btn_capture.setStyleSheet(_BTN_STYLE)
        self.btn_capture.clicked.connect(self._on_capture)

        sb.addWidget(self.lbl_title)
        sb.addWidget(lbl_hint)
        sb.addStretch()
        sb.addWidget(self.btn_capture)
        root.addWidget(sidebar)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def reset(self):
        """Reset state for a new capture session."""
        self._stage = 0
        self._rgb_closed = None
        self._depth_closed = None
        self._bbox_closed_xywh = None
        self._lmk2d_closed = None
        self._lmk3d_closed = None
        self._lmk_mask_closed = None
        self._aruco_cache = None
        self._aruco_age = 0
        self._frame_buffer = []
        self._preview_bbox = None
        self._frame_count = 0
        self.lbl_title.setText("采集闭口图像")
        self.btn_capture.setText("采集闭口")
        self.btn_capture.setEnabled(True)
        self._reset_preview_labels()

    def on_frame(self, rgb: np.ndarray, depth: np.ndarray, camera):
        """Called by CameraThread.frame_ready signal."""
        if self._warning_active:
            return

        self._camera = camera
        self._rgb    = rgb
        self._depth  = depth
        self._frame_buffer.append((rgb.copy(), depth.copy()))
        self._frame_buffer = self._frame_buffer[-3:]
        self._frame_count += 1

        if self._frame_count % self._BBOX_PREVIEW_INTERVAL == 0:
            self._bbox_thread.submit(rgb)

        if self._stage == 0:
            # 闭口阶段只刷新上排，下排保持占位底色。
            self.lbl_closed_rgb.set_image(self._overlay_preview(rgb, with_aruco=True))
            if self._frame_count % self._DEPTH_PREVIEW_INTERVAL == 0:
                self.lbl_closed_depth.set_image(depth_to_rgb(depth))
        else:
            # 张口阶段上排固定为闭口采集结果，只刷新下排。
            self.lbl_open_rgb.set_image(self._overlay_preview(rgb, with_aruco=False))
            if self._frame_count % self._DEPTH_PREVIEW_INTERVAL == 0:
                self.lbl_open_depth.set_image(depth_to_rgb(depth))

    def _reset_preview_labels(self):
        """恢复四宫格占位文本，避免重新采集时残留上一轮图片。"""
        for label, text in (
            (self.lbl_closed_rgb, "闭口 RGB"),
            (self.lbl_closed_depth, "闭口 Depth"),
            (self.lbl_open_rgb, "张口 RGB"),
            (self.lbl_open_depth, "张口 Depth"),
        ):
            label.clear()
            label.setText(text)

    def _on_bbox_ready(self, bbox):
        if self._warning_active:
            return
        self._preview_bbox = bbox

    def _overlay_preview(self, rgb: np.ndarray, with_aruco: bool = False) -> np.ndarray:
        """在实时 RGB 预览中叠加 YuNet 人脸框，可选叠加 ArUco 框。"""
        vis = self._overlay_aruco(rgb) if with_aruco else rgb.copy()
        return self._draw_face_bbox(vis, self._preview_bbox)

    @staticmethod
    def _draw_face_bbox(vis: np.ndarray, bbox) -> np.ndarray:
        if bbox is None:
            return vis

        x, y, w, h = [int(v) for v in bbox]
        color = (0, 255, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, "Face", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return vis

    def _overlay_aruco(self, rgb: np.ndarray) -> np.ndarray:
        """在 RGB 图上检测并绘制 ArUco 绿框，带缓存防闪烁。"""
        should_detect = self._frame_count % self._ARUCO_PREVIEW_INTERVAL == 0
        if should_detect:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners_list, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)

            if ids is not None:
                self._aruco_cache = (corners_list, ids)
                self._aruco_age = 0
            else:
                self._aruco_age += 1

        if self._aruco_cache is None or self._aruco_age > self._ARUCO_HOLD_FRAMES:
            self._aruco_cache = None
            return rgb

        cached_corners, cached_ids = self._aruco_cache
        vis = rgb.copy()
        for corners, mid in zip(cached_corners, cached_ids):
            pts = corners[0].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            center = corners[0].mean(axis=0).astype(np.int32)
            cv2.circle(vis, (center[0], center[1]), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"ID:{mid[0]}", (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return vis

    # ------------------------------------------------------------------
    # internal slots
    # ------------------------------------------------------------------
    def _on_capture(self):
        if self._rgb is None or self._depth is None or self._camera is None:
            return
        if len(self._frame_buffer) < 3:
            self._show_warning("正在缓存三帧图像，请稍后再采集")
            return

        import numpy as np

        capture_frames = self._frame_buffer[-3:]
        rgb_capture = capture_frames[0][0]
        depth_capture = merge_depth_frames([frame[1] for frame in capture_frames])

        # Step 2: 人脸检测 (YuNet → [x, y, w, h])
        bbox_xywh = detect_face_bbox(rgb_capture)
        if bbox_xywh is None:
            self._show_warning("未检测到人脸，请重新采集")
            return

        if self._stage == 0:
            # Step 3: ArUco 喉结检测（仅闭口阶段）
            corners_c, _, _ = detect_aruco(rgb_capture, depth_capture, self._camera)
            if corners_c is None:
                self._show_warning("未检测到喉结，请重新拍摄")
                return

        # Step 4: TUFA 关键点（YuNet xywh bbox 直接传入，格式与 get_transforms 一致）
        lmk2d = get_lmk_single(rgb_capture, bbox_xywh)
        lmk3d, mask = backproject_window(
            np.asarray(lmk2d, dtype=np.float32),
            depth_capture,
            self._camera,
            window=2,
            trim=0,
            keep_shape=True,
        )
        align_mask = mask[17:48].copy()
        align_mask[_ALIGN_EXCLUDED_LMK_68 - 17] = False
        if int(align_mask.sum()) < _MIN_ALIGN_LANDMARKS:
            self._show_warning(
                f"可用于对齐的关键点深度不足，请重新采集 "
                f"({int(align_mask.sum())}/{_ALIGN_LMK_COUNT - len(_ALIGN_EXCLUDED_LMK_68)})"
            )
            return
        missing_required = _REQUIRED_LMK_68[~mask[_REQUIRED_LMK_68]]
        if missing_required.size > 0:
            self._show_warning(
                "关键语义点深度无效，请重新采集 "
                f"({', '.join(str(int(i)) for i in missing_required)})"
            )
            return
        lmk3d = coordinate_transform(lmk3d)

        if self._stage == 0:
            self._rgb_closed       = rgb_capture.copy()
            self._depth_closed     = depth_capture.copy()
            self._bbox_closed_xywh = bbox_xywh.copy()
            self._lmk2d_closed     = np.asarray(lmk2d, dtype=np.float32)
            self._lmk3d_closed     = lmk3d.copy()
            self._lmk_mask_closed  = mask.copy()
            closed_vis = self._overlay_aruco(self._rgb_closed)
            self.lbl_closed_rgb.set_image(self._draw_face_bbox(closed_vis, self._bbox_closed_xywh))
            self.lbl_closed_depth.set_image(depth_to_rgb(self._depth_closed))
            self._preview_bbox = None
            self.lbl_open_rgb.clear()
            self.lbl_open_rgb.setText("张口 RGB")
            self.lbl_open_depth.clear()
            self.lbl_open_depth.setText("张口 Depth")
            self._stage = 1
            self._aruco_cache = None
            self._aruco_age = 0
            self._frame_buffer = []
            self.lbl_title.setText("采集张口图像")
            self.btn_capture.setText("采集张口")
        else:
            self.btn_capture.setEnabled(False)
            landmarks = {
                "lmk2d_closed": self._lmk2d_closed,
                "lmk2d_open": np.asarray(lmk2d, dtype=np.float32),
                "lmk3d_closed": self._lmk3d_closed,
                "lmk3d_open": lmk3d.copy(),
                "mask_closed": self._lmk_mask_closed,
                "mask_open": mask.copy(),
            }
            self.capture_done.emit(
                self._rgb_closed,    rgb_capture.copy(),
                self._depth_closed,  depth_capture.copy(),
                self._camera,
                self._bbox_closed_xywh, bbox_xywh.copy(),
                landmarks,
            )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._on_capture()
        else:
            super().keyPressEvent(event)

    def _show_warning(self, message: str):
        """弹窗期间暂停预览处理，避免 modal 事件循环堆积相机帧。"""
        self._warning_active = True
        self.btn_capture.setEnabled(False)
        self._frame_buffer = []
        self._preview_bbox = None
        try:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("提示")
            box.setText(message)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setStyleSheet("""
                QMessageBox {
                    background: #ffffff;
                }
                QMessageBox QLabel {
                    color: #212121;
                    font-size: 14px;
                }
                QMessageBox QPushButton {
                    background: #1976d2;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    min-width: 72px;
                    min-height: 28px;
                    padding: 4px 12px;
                }
                QMessageBox QPushButton:hover {
                    background: #1565c0;
                }
            """)
            box.exec()
        finally:
            self.btn_capture.setEnabled(True)
            self._warning_active = False
            self.activateWindow()
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    def closeEvent(self, event):
        self._bbox_thread.stop()
        event.accept()

    def stop_threads(self):
        self._bbox_thread.stop()
