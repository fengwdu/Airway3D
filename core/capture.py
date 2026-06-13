import numpy as np
from pyorbbecsdk import (
    Pipeline, Config,
    OBSensorType, OBFormat, OBAlignMode,
    OBFrameAggregateOutputMode, OBDepthWorkMode,
)
from core.camera import Camera
from core.config import DEPTH_VMIN_MM, DEPTH_VMAX_MM, DEPTH_WORK_MODE


# -----------------------------------------------------------------------
# 私有辅助函数
# -----------------------------------------------------------------------

def _select_rgb_1280x720(pipeline: Pipeline):
    """从设备支持的彩色 profile 列表中筛选 RGB 格式, 分辨率 1280x720."""
    plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if plist is None or len(plist) == 0:
        raise RuntimeError("No COLOR stream profiles found.")

    for i in range(len(plist)):
        p = plist[i]
        if p.get_format() == OBFormat.RGB and p.get_width() == 1280 and p.get_height() == 720:
            return p

    raise RuntimeError("No RGB 1280x720 color profile found.")


def _get_hw_d2c_config(pipeline: Pipeline, color_profile):
    """
    构建硬件 D2C (Depth-to-Color) 对齐配置.
    - 深度流在设备端直接对齐到彩色流, 无需软件对齐, 延迟更低
    - 按目标分辨率匹配深度 profile, 避免硬编码索引在不同固件版本下失效
    """
    hw_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    if hw_list is None or len(hw_list) == 0:
        raise RuntimeError("HW D2C not supported for this color profile/device.")

    # 从 hw_list 中选择最优深度 profile:
    # 优先条件: Y16 格式 (SDK 标准输出格式) + 分辨率尽量大 + 帧率尽量高
    # 实测 hw_list 只有 1280x800 和 640x400 两种分辨率, 没有与彩色流完全对齐的尺寸,
    # D2C 后深度图会被裁剪/缩放到与彩色流相同的尺寸
    best = None
    best_score = (-1, -1, -1)   # (w*h, fps, is_y16)
    for i in range(len(hw_list)):
        p = hw_list[i]
        is_y16 = 1 if p.get_format() == OBFormat.Y16 else 0
        score = (p.get_width() * p.get_height(), p.get_fps(), is_y16)
        if score > best_score:
            best_score = score
            best = p
    depth_profile = best if best is not None else hw_list[0]

    config = Config()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_align_mode(OBAlignMode.HW_MODE)

    try:
        # 要求彩色帧和深度帧必须同时到达才输出, 避免时间戳不同步
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception:
        pass

    return config


# -----------------------------------------------------------------------
# 深度工作模式管理
# 设备支持两种模式:
#   - "Close_Up Precision Mode"  : 近距离精度模式
#   - "Extended Distance Mode"   : 远距离模式 (本项目使用)
# -----------------------------------------------------------------------

def list_depth_work_modes() -> list[OBDepthWorkMode]:
    """查询设备支持的所有深度工作模式, 返回模式对象列表."""
    pipeline = Pipeline()
    device = pipeline.get_device()
    mode_list = device.get_depth_work_mode_list()
    return [mode_list.get_depth_work_mode_by_index(i) for i in range(mode_list.get_count())]


def get_depth_work_mode() -> OBDepthWorkMode:
    """返回设备当前的深度工作模式."""
    pipeline = Pipeline()
    device = pipeline.get_device()
    return device.get_depth_work_mode()


def set_depth_work_mode(mode: OBDepthWorkMode | str) -> OBDepthWorkMode:
    """
    切换设备深度工作模式.

    参数:
        mode: OBDepthWorkMode 对象, 或模式名称字符串
              (例如 "Extended Distance Mode")
    返回:
        切换后实际生效的 OBDepthWorkMode 对象
    异常:
        ValueError: 名称字符串未匹配到任何可用模式时抛出
    """
    pipeline = Pipeline()
    device = pipeline.get_device()

    # 如果传入的是字符串, 先在模式列表中按名称查找对应对象
    if isinstance(mode, str):
        mode_list = device.get_depth_work_mode_list()
        target = None
        for i in range(mode_list.get_count()):
            m = mode_list.get_depth_work_mode_by_index(i)
            if m.name == mode:
                target = m
                break
        if target is None:
            available = [mode_list.get_depth_work_mode_by_index(i).name
                         for i in range(mode_list.get_count())]
            raise ValueError(f"Depth work mode {mode!r} not found. Available: {available}")
        mode = target

    device.set_depth_work_mode(mode.name)
    return device.get_depth_work_mode()


# -----------------------------------------------------------------------
# 流式采集器
# -----------------------------------------------------------------------

class OrbbecCapture:
    """
    Orbbec 相机流式采集器.

    典型用法:
        cap = OrbbecCapture(timeout_ms=100)
        cap.start()
        while True:
            out = cap.read()
            if out is None:
                continue
            camera, rgb, depth = out
        cap.stop()

    read() 返回值:
        (Camera, rgb, depth) 或 None (帧未就绪时)
        - rgb   : (720, 1280, 3) uint8, RGB 格式
        - depth : (720, 1280)    uint16, 单位 mm
    """

    def __init__(self, timeout_ms: int = 100):
        self.timeout_ms = timeout_ms
        self.pipeline = Pipeline()
        self.camera = None       # 首帧到达后由内参初始化
        self._started = False

    def start(self):
        # 步骤 1: 在启动流之前切换到远距离深度工作模式
        # 官方文档要求: Depth mode 必须在 start stream 之前设置
        # 官方示例传入模式名称字符串 (device.set_depth_work_mode(mode.name))
        device = self.pipeline.get_device()
        device.set_depth_work_mode(DEPTH_WORK_MODE)

        # 步骤 2: 选择彩色流 profile (RGB 1280x720)
        color_profile = _select_rgb_1280x720(self.pipeline)

        # 步骤 3: 构建 HW D2C 配置并启动 pipeline
        config = _get_hw_d2c_config(self.pipeline, color_profile)
        self.pipeline.start(config)
        self._started = True

    def stop(self):
        # 只停止 pipeline, 窗口生命周期由调用方管理
        if self._started:
            self.pipeline.stop()
            self._started = False

    def read(self):
        """
        读取一帧, 返回 (camera, rgb, depth) 或 None.
        - 帧超时/未就绪/格式不符时均返回 None, 调用方应循环重试
        """
        if not self._started:
            raise RuntimeError("OrbbecCapture.start() must be called first.")

        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        # HW D2C 对齐后, 彩色帧和深度帧应都是 1280x720
        if color_frame.get_width() != 1280 or color_frame.get_height() != 720:
            return None
        if depth_frame.get_width() != 1280 or depth_frame.get_height() != 720:
            return None

        # 格式校验
        if color_frame.get_format() != OBFormat.RGB:
            return None
        if depth_frame.get_format() != OBFormat.Y16:
            return None

        # 将原始字节数据解析为 numpy 数组
        rgb   = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape((720, 1280, 3))
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((720, 1280))

        # 首帧成功时从彩色流内参初始化相机模型
        if self.camera is None:
            intr = color_frame.get_stream_profile().get_intrinsic()
            self.camera = Camera(float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy))

        return self.camera, rgb, depth
