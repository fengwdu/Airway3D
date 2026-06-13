"""
项目级常量与超参配置
"""
import sys
from pathlib import Path


"""
目录部分
"""

# 运行时_MEIPASS目录
if getattr(sys, 'frozen', False):
    RUN_DIR = Path(sys._MEIPASS)
else:
    RUN_DIR = Path(__file__).parent.parent

# 项目目录
if getattr(sys, 'frozen', False):
    PROJECT_DIR = Path(sys.executable).parent
else:
    PROJECT_DIR = Path(__file__).parent.parent

ASSETS_DIR = PROJECT_DIR / "assets"
OUTPUT_DIR = PROJECT_DIR / "output"

"""
相机部分
"""

DEPTH_WORK_MODE = "Extended Distance Mode"

DEPTH_VMIN_MM = 200
DEPTH_VMAX_MM = 700

"""
bbox识别部分
"""

bbox_model_path = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
score_threshold = 0.6  # 置信度阈值
nms_threshold = 0.3  # NMS 阈值
top_k = 5000  # NMS 前保留候选框数量