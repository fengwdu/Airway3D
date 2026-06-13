from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox

from .capture_widget    import CaptureWidget
from .processing_widget import ProcessingWidget
from .result_widget     import ResultWidget
from .workers           import CameraThread, PipelineWorker

PAGE_CAPTURE    = 0
PAGE_PROCESSING = 1
PAGE_RESULT     = 2


class MainWindow(QMainWindow):
    """
    Top-level window. Holds three pages (QStackedWidget) and two background threads.

    Data flow:
        CameraThread  --frame_ready-->  CaptureWidget.on_frame
        CaptureWidget --capture_done--> _on_capture_done
            stop camera -> switch to page 2 -> start PipelineWorker
        PipelineWorker --progress-->  ProcessingWidget.set_status
        PipelineWorker --finished-->  _on_finished -> switch to page 3
        ResultWidget   --restart-->   _on_restart  -> reset and back to page 1
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airway3D")
        self.resize(1400, 820)
        self.setStyleSheet("background: #f5f5f5;")

        # pages
        self._stack   = QStackedWidget()
        self._capture = CaptureWidget()
        self._process = ProcessingWidget()
        self._result  = ResultWidget()
        self._stack.addWidget(self._capture)   # PAGE_CAPTURE
        self._stack.addWidget(self._process)   # PAGE_PROCESSING
        self._stack.addWidget(self._result)    # PAGE_RESULT
        self.setCentralWidget(self._stack)

        # signal connections
        self._capture.capture_done.connect(self._on_capture_done)
        self._result.restart.connect(self._on_restart)

        self._pipeline: PipelineWorker | None = None

        # start camera
        self._start_camera()

    # ------------------------------------------------------------------
    # camera management
    # ------------------------------------------------------------------
    def _start_camera(self):
        self._cam_thread = CameraThread()
        self._cam_thread.frame_ready.connect(self._capture.on_frame)
        self._cam_thread.start()

    def _stop_camera(self):
        self._cam_thread.frame_ready.disconnect()
        self._cam_thread.stop()

    # ------------------------------------------------------------------
    # page transition slots
    # ------------------------------------------------------------------
    def _on_capture_done(self, rgb_c, rgb_o, dep_c, dep_o, camera, bbox_c, bbox_o, landmarks):
        # 采集页完成后：
        # - 停止实时预览线程，冻结本次采集数据
        # - 切到处理中页面
        # - 启动后台 PipelineWorker 执行完整算法链
        self._stop_camera()
        self._rgb_closed = rgb_c
        self._rgb_open   = rgb_o

        self._process.setup(rgb_c, rgb_o, dep_c, dep_o)
        self._stack.setCurrentIndex(PAGE_PROCESSING)

        self._pipeline = PipelineWorker(rgb_c, rgb_o, dep_c, dep_o, camera,
                                        bbox_closed_xywh=bbox_c, bbox_open_xywh=bbox_o,
                                        landmarks=landmarks)
        self._pipeline.progress.connect(self._process.set_status)
        self._pipeline.finished.connect(self._on_finished)
        self._pipeline.error.connect(self._on_error)
        self._pipeline.start()

    def _on_finished(self, metrics: dict, proj_closed, proj_open, aruco_corners):
        # 处理完成后：
        # - 先把检测到的 ArUco 叠加到处理中页面用于过程回看
        # - 再把指标 + 渲染结果传给结果页显示
        self._process.draw_aruco(aruco_corners)
        self._result.setup(metrics, proj_closed, proj_open,
                           self._rgb_closed, self._rgb_open, aruco_corners)
        self._stack.setCurrentIndex(PAGE_RESULT)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "处理失败", msg)
        self._on_restart()

    def _on_restart(self):
        # 用户点击“重新采集”：清理采集页状态并回到首页，重新拉起相机线程
        self._capture.reset()
        self._stack.setCurrentIndex(PAGE_CAPTURE)
        self._start_camera()

    # ------------------------------------------------------------------
    # cleanup on close
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._cam_thread.stop()
        self._capture.stop_threads()
        if self._pipeline and self._pipeline.isRunning():
            self._pipeline.terminate()
            self._pipeline.wait()
        event.accept()
