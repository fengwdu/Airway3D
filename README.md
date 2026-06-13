# Airway3D 项目说明

Airway3D 是一个基于 RGB-D 相机的三维气道评估原型系统。项目通过 Orbbec 深度相机采集闭口与张口两组 RGB/Depth 图像，结合人脸检测、关键点检测、人脸/颈部分割、FLAME 三维头面模型拟合和 ArUco 喉结定位，输出闭口/张口三维网格以及一组气道相关评估指标。


## 功能概览

- 实时采集 Orbbec RGB-D 图像，硬件 D2C 对齐到 1280x720。
- 采集闭口、张口两种状态，并合并连续三帧深度图以降低空洞噪声。
- 使用 YuNet 检测人脸框。
- 使用 TUFA 模型检测 68 点人脸关键点。
- 使用 BiSeNet 人脸解析模型提取面部与颈部点云。
- 使用 ArUco 标记定位喉结点，作为甲颏距离等指标的锚点。
- 使用 FLAME 模型拟合闭口/张口三维头面网格。
- 计算并展示张口度、甲颌距离、头后仰角度、下颌活动角度、下颌活动度、颈围、舌厚度、声门视野等指标。
- 在 PyQt6 桌面界面中显示采集、处理进度、拟合网格和评估结果。

## 项目结构

```text
Airway3D/
├── main.py                         # PyQt6 应用入口
├── flame_explorer.py               # FLAME 参数交互式可视化工具
├── requirements.txt                # Python 依赖列表
├── assets/                         # 模型权重、相机配置和 FLAME 资源
├── core/                           # 核心算法与相机逻辑
│   ├── capture.py                  # Orbbec 相机采集与 D2C 对齐
│   ├── camera.py                   # 相机模型、深度反投影、坐标转换
│   ├── face_test.py                # YuNet 人脸检测
│   ├── lmkdetection.py             # 关键点相关逻辑
│   ├── mask_detection.py           # 人脸/颈部分割与点云生成
│   ├── aruco.py                    # ArUco 检测与喉结 3D 定位
│   ├── fitting.py                  # FLAME 双状态拟合
│   ├── metrics.py                  # 评估指标计算
│   └── render.py                   # 三维网格离屏渲染
├── ui/                             # PyQt6 界面
│   ├── main_window.py              # 页面切换与线程调度
│   ├── capture_widget.py           # 采集页
│   ├── processing_widget.py        # 处理进度页
│   ├── result_widget.py            # 结果页
│   └── workers.py                  # 相机线程与算法流水线线程
├── tests/                          # 调试与实验脚本
└── third_party/                    # FLAME、TUFA 等第三方代码
```

## 运行环境

建议环境：

- Windows
- Python 3.10
- NVIDIA GPU 与 CUDA 环境，推荐用于 PyTorch/FLAME 拟合加速
- Orbbec RGB-D 相机
- 可识别的 `DICT_4X4_50` ArUco 标记，用于闭口图像中的喉结定位

主要依赖见 `requirements.txt`，包括：

- `PyQt6`
- `opencv-python`
- `numpy`
- `torch` / `torchvision`
- `trimesh`
- `pyrender`
- `smplx`
- `uniface`
- `batch-face`
- `pyorbbecsdk2`

`flame_explorer.py` 额外依赖 `pyvista`，该依赖当前没有写在 `requirements.txt` 中。

## 安装依赖

推荐使用独立虚拟环境：

```powershell
conda create -n airway3d python=3.10
conda activate airway3d
pip install -r requirements.txt
```

如果需要运行 FLAME 参数浏览器：

```powershell
pip install pyvista
```

项目的 `requirements.txt` 已配置 PyTorch CUDA 轮子的额外索引：

```text
--extra-index-url https://download.pytorch.org/whl/cu130
```

如果本机 CUDA 或 PyTorch 版本不匹配，请根据实际 GPU 驱动和 CUDA 版本调整 PyTorch 安装方式。

## 模型与资源文件

项目运行依赖 `assets/` 和 `third_party/tufa/weights/` 中的模型资源。当前代码会读取或使用以下关键文件：

```text
assets/
├── face_detection_yunet_2023mar.onnx
├── flame2023.pkl
├── generic_model.pkl
├── flame_static_embedding.pkl
├── flame_dynamic_embedding.npy
├── verts2face.pt
├── adjacency.pt
└── camera.json

third_party/tufa/weights/
└── Face_prompt_base.pth
```

其中 `flame2023.pkl` 来源于 FLAME 官方模型授权下载。请确保模型文件已按授权要求放置在 `assets/` 目录下。

## 启动应用

连接 Orbbec 相机后，在项目根目录运行：

```powershell
python main.py
```

启动时程序会先预热两类模型：

1. 人脸分割模型：`core.mask_detection.get_parser()`
2. TUFA 关键点模型：`third_party.tufa.Test._get_tufa()`

加载完成后进入主窗口。

## 使用流程

1. 将 Orbbec 相机对准受试者头颈区域。
2. 确保闭口采集阶段画面中可见 ArUco 标记。项目会用该标记中心点作为喉结 3D 锚点。
3. 在首页点击“采集闭口”或按空格键。
4. 保持张口状态，再点击“采集张口”或按空格键。
5. 程序进入处理页，依次完成关键点读取、ArUco 检测、面颈分割、点云整理、FLAME 拟合、指标计算和结果渲染。
6. 结果页展示原图、拟合网格和评估指标。
7. 点击“重新采集”可开始下一轮采集。

## 处理流水线

主流程由 `ui.workers.PipelineWorker` 串联：

```text
闭口/张口 RGB-D
    ↓
复用采集阶段通过质检的 TUFA 2D/3D 关键点
    ↓
闭口图像检测 ArUco，反投影得到喉结 3D 点
    ↓
BiSeNet 分割面部与颈部区域
    ↓
深度反投影生成 face 与 face+neck 点云
    ↓
FLAME 双状态拟合
    ↓
导出 mesh_closed.obj / mesh_open.obj
    ↓
计算指标
    ↓
离屏渲染网格并显示结果
```

## 输出结果

处理完成后，程序会导出拟合网格：

```text
output/
├── closed/
│   └── mesh_closed.obj
└── open/
    └── mesh_open.obj
```

界面中展示的指标包括：

- 张口度
- 甲颌距离
- 头后仰角度
- 下颌活动角度
- 下颌活动度
- 颈围
- 舌厚度
- 声门视野

指标计算逻辑位于 `core/metrics.py`。其中部分点位来自 FLAME 网格顶点索引，部分点位来自 TUFA 68 点关键点和 ArUco 喉结点。

## 调试脚本

### FLAME 参数浏览器

用于观察 shape、expression、pose 参数对 FLAME 网格的影响：

```powershell
python flame_explorer.py
```

该脚本需要 `pyvista`，并要求 `assets/flame2023.pkl` 可用。

### 人脸解析调试

对保存的 RGB 图像执行全图解析与裁剪区域解析，并输出 mask、overlay 和 label 统计：

```powershell
python tests/parse_capture_rgb.py --image tests/flame_fit_face_neck_output/capture_rgb.jpg --out-dir tests/parse_capture_rgb_output
```

### FLAME 面颈拟合实验

`tests/flame_fit_face_neck_cv.py` 是面部/颈部点云、关键点和 FLAME 拟合的实验脚本，可用于单独验证相机采集、分割和拟合链路。

## 常见问题

### 1. 启动后没有相机画面

- 确认 Orbbec 相机已连接并能被 SDK 识别。
- 确认设备支持 RGB 1280x720 彩色流。
- 确认设备支持硬件 D2C 对齐。
- 检查 `pyorbbecsdk2` 是否安装成功。

### 2. 提示未检测到人脸

- 保证脸部在画面中央且光照充足。
- 避免过大的头部旋转、遮挡或离相机过远。
- 检查 `assets/face_detection_yunet_2023mar.onnx` 是否存在。

### 3. 提示未检测到喉结或 ArUco

- 闭口采集阶段必须能看到 ArUco 标记。
- 使用 OpenCV `DICT_4X4_50` 字典对应的标记。
- 保证标记边缘清晰、无遮挡、不过曝。

### 4. 拟合或渲染很慢

- 优先使用 GPU 环境运行 PyTorch。
- 检查 PyTorch CUDA 是否可用。
- 离屏渲染依赖 `pyrender`，部分机器可能需要额外配置 OpenGL/EGL 环境。

### 5. `flame_explorer.py` 无法启动

- 安装 `pyvista`。
- 确认 `assets/flame2023.pkl` 存在。
- 如果 FLAME pickle 与当前 NumPy 版本存在兼容问题，脚本内已有 chumpy stub，但仍建议使用项目推荐的 Python 3.10 环境。

## 开发备注

- 相机深度单位为毫米，反投影后转换为米。
- `core.camera.coordinate_transform()` 会将相机坐标系转换为模型坐标系：Y 轴取反、Z 轴取反。
- `CaptureWidget` 在闭口与张口采集时会分别检测人脸、关键点，并做 3D landmark 反投影质检。
- `PipelineWorker` 复用采集阶段已经通过质检的关键点，避免处理阶段重复检测造成不一致。
- FLAME 拟合结果使用 `trimesh` 导出为 OBJ。
- 当前指标单位在 UI 中主要显示为厘米和角度。
