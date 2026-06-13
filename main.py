import os
import sys

if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QFont, QColor
from PyQt6.QtCore import Qt


def main():
    # 应用启动入口：
    # 1) 创建 Qt 应用和启动页
    # 2) 预热推理模型（分割/关键点）
    # 3) 进入主窗口（采集 -> 处理 -> 结果）
    app = QApplication(sys.argv) # 创建 Qt 应用
    app.setStyle("Fusion") # 设置样式

    pixmap = QPixmap(420, 160) # 创建启动页图片
    pixmap.fill(QColor("#f5f5f5")) # 填充颜色
    splash = QSplashScreen(pixmap) # 创建启动页
    splash.setFont(QFont("Microsoft YaHei", 13)) # 设置字体
    splash.show() # 显示启动页

    def status(msg):
        splash.showMessage(msg, Qt.AlignmentFlag.AlignCenter, QColor("#333333"))
        app.processEvents()

    status("正在加载人脸分割模型...")
    # 预热 face parsing（core/mask_detection.py）
    from core.mask_detection import get_parser
    get_parser()



    status("正在加载关键点模型...")
    # 预热 TUFA 关键点模型（third_party/tufa/Test.py）
    from third_party.tufa.Test import _get_tufa
    _get_tufa()

    splash.close()

    # UI 主窗口负责串联完整业务流程
    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
