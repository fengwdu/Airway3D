# Airway3D 项目说明

Airway3D 是一个基于 RGB-D 相机的三维气道评估原型系统。项目通过 Orbbec 深度相机采集闭口与张口两组 RGB/Depth 图像，结合人脸检测、关键点检测、人脸/颈部分割、FLAME 三维头面模型拟合和 ArUco 喉结定位，输出闭口/张口三维网格以及一组气道相关评估指标。

## FLAME 模型说明

本项目使用 `third_party/flame/` 中的 PyTorch 版 FLAME 头面模型。FLAME 是一个可微的三维统计头部模型，核心思想是用少量参数控制头面网格：

- `shape_params`：身份/头面形状参数，描述个体差异。
- `expression_params`：表情参数，描述嘴唇、面颊等软组织表情变化。
- `pose_params`：关节姿态参数，本项目配置为 6 维，对应颈部与下颌姿态。
- `head_pose`：全局头部姿态参数。
- `transl` / `trans_params`：三维平移参数。

`third_party/flame/config.py` 中的默认配置为：

```python
shape_params = 50
expression_params = 10
pose_params = 6
batch_size = 2
```

其中 `batch_size = 2` 对应闭口、张口两帧同时拟合。模型文件路径包括：

```text
assets/flame2023.pkl
assets/flame_static_embedding.pkl
assets/flame_dynamic_embedding.npy
```

`third_party/flame/flame.py` 的 `FLAME.forward()` 接收 shape、expression、pose 和 translation，内部通过 SMPL-X 的 LBS 逻辑生成：

```text
vertices:  (N, V, 3)  # FLAME 网格顶点
landmarks: (N, K, 3)  # 由网格面片 barycentric 插值得到的 3D landmarks
```

代码中会把 identity shape、expression、head/neck/jaw/eye pose 拼成 FLAME 所需的 `betas` 和 `full_pose`，再调用 `smplx.lbs.lbs()` 得到变形后的网格。静态 landmark 来自 `flame_static_embedding.pkl`，动态脸轮廓 landmark 来自 `flame_dynamic_embedding.npy`；启用动态轮廓时，FLAME 会根据颈部/头部旋转选择对应的轮廓面片与重心坐标。

在本项目中，`core/fitting.py` 使用 FLAME 的方式如下：

1. 先用闭口、张口两帧的 TUFA 3D landmark 做一次 Umeyama 刚性对齐，将 FLAME 初始 landmark 放到 RGB-D 观测点云坐标系中。
2. 使用 face+neck 点云作为主要几何约束，对 FLAME 顶点做 point-to-point 与 point-to-plane 拟合。
3. Stage1 只优化头部/颈部/下颌相关姿态和平移，并使用外唇 3D 距离约束做粗稳定。
4. Stage2 优化 shape、expression、pose、head_pose 与 trans，并加入语义 3D landmark 点约束和内唇照片平面距离约束。
5. 拟合完成后导出闭口、张口两个 `trimesh.Trimesh` 网格，并把拟合后的 FLAME landmarks 存入 mesh metadata 供指标计算使用。


## FLAME 拟合与损失函数

拟合逻辑位于 `core/fitting.py`。主函数 `train()` 同时处理闭口、张口两帧，因此 FLAME 配置中的 `batch_size = 2` 分别对应：

```text
b = 0: 闭口帧
b = 1: 张口帧
```

进入优化前，代码会先准备三类固定目标，避免在每个 Adam step 中重复计算：

- `mouth_3d_target`：Stage1 使用的外唇 3D 距离目标。
- `semantic_target`：Stage2 使用的语义 3D landmark 目标点。
- `mouth_image_target`：Stage2 使用的内唇照片平面距离目标。

FLAME 初始 landmark 会先通过 TUFA 3D landmark 做一次 Umeyama 刚性对齐，得到每帧的 `rotations` 和 `translations`，把 FLAME 网格放到 RGB-D 观测坐标系中。后续优化中的 FLAME 顶点和 landmark 都在这个观测坐标系下计算。

当前总损失形式为：

```text
0.5 * (
  point_cloud
+ point_plane
+ mouth_3d_distance      # 仅 Stage1 打开
+ mouth_image_distance   # 仅 Stage2 打开
+ semantic_landmark      # 仅 Stage2 打开
+ shape_reg
+ expr_reg
)
```

各项权重在 `train()` 中设置：

```python
point_cloud = 0.1
point_plane = 0.9
mouth_distance = 10.0
mouth_image_distance = 1e-5
landmark_points = 2.0
shape_reg = 1e-6
expr_reg = 1e-6
```

点云损失是拟合的主体。每个 outer iteration 会重新为采样后的 face+neck 点云寻找最近 FLAME 顶点，并冻结当前顶点法线；inner iteration 中使用这些对应关系计算 point-to-point 与 point-to-plane loss。

Stage1 使用：

```text
outer_iters = 6
inner_iters = 120
optimizer lr = 5e-3
```

此阶段只优化 `pose_params`、`head_pose` 和 `trans_params`，并通过梯度 mask 只允许 `pose_params[0]` 与 `pose_params[3]` 更新，也就是颈部第一个姿态分量和下颌第一个姿态分量。Stage1 打开 `mouth_3d_distance`，使用外唇三对 68 点 `(50,58)、(51,57)、(52,56)`，让 FLAME 对应 landmark 的 3D 距离跟随闭口/张口各自反投影得到的 3D 外唇距离，用于粗稳定。

Stage2 使用：

```text
outer_iters = 12
inner_iters = 160
optimizer lr = 1e-2
```

此阶段优化 `shape_params`、`expr_params`、`pose_params`、`head_pose` 和 `trans_params`，但 pose 仍通过梯度 mask 只允许 `pose_params[0]` 与 `pose_params[3]` 更新。Stage2 关闭外唇 3D 距离，打开两类额外约束：

- `semantic_landmark`：使用 `[8,21,22,33,50,51,52,56,57,58]` 这些语义 landmark，将 FLAME landmark 拉向反投影得到的 3D 语义点。
- `mouth_image_distance`：使用内唇三对 68 点 `(61,67)、(62,66)、(63,65)`。代码先把 FLAME fitted landmark 从观测坐标系转回相机坐标系，再调用 `backprojection_inverse()` 投影到照片坐标，约束其像素距离接近 TUFA 在照片中检测出的内唇像素距离。

`tol_loss = 1e-9` 用于 inner loop 的早停判断。当相邻两步 loss 变化小于该阈值时，当前 inner loop 会提前结束。

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
- CUDA 环境
- Orbbec RGB-D 相机
- 可识别的 `DICT_4X4_50` ArUco 标记，用于闭口图像中的喉结定位


## 安装依赖

推荐使用独立虚拟环境：

```powershell
conda create -n airway3d python=3.10
conda activate airway3d
pip install -r requirements.txt
pip install chumpy --no-build-isolation
```

`chumpy` 用于兼容 FLAME pickle 相关依赖，建议按上面的单独命令安装；不要让它走默认 build isolation，否则在部分环境中可能构建失败。

相机配置请参考 [text](https://orbbec.github.io/pyorbbecsdk/index.html)

如果需要运行 FLAME 参数浏览器（FLAME 参数可视化，与项目无直接关系）：

```powershell
pip install pyvista
```

项目的 `requirements.txt` 已配置 PyTorch CUDA 的额外索引：

```text
--extra-index-url https://download.pytorch.org/whl/cu130
```

如果本机 CUDA 或 PyTorch 版本不匹配，需要根据实际 GPU 驱动和 CUDA 版本调整 PyTorch 安装方式。

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

### 提示未检测到人脸

- 保证脸部在画面中央且光照充足。
- 避免过大的头部旋转、遮挡或离相机过远。
- 检查 `assets/face_detection_yunet_2023mar.onnx` 是否存在。

### 提示未检测到喉结或 ArUco

- 闭口采集阶段必须能看到 ArUco 标记。
- 使用 OpenCV `DICT_4X4_50` 字典对应的标记。
- 保证标记边缘清晰、无遮挡、不过曝。

## 开发备注

- 相机深度单位为毫米，反投影后转换为米。
- `core.camera.coordinate_transform()` 会将相机坐标系转换为模型坐标系：Y 轴取反、Z 轴取反。
- `CaptureWidget` 在闭口与张口采集时会分别检测人脸、关键点，并做 3D landmark 反投影质检。
- `PipelineWorker` 复用采集阶段已经通过质检的关键点，避免处理阶段重复检测造成不一致。
- FLAME 拟合结果使用 `trimesh` 导出为 OBJ。
- 当前指标单位在 UI 中主要显示为厘米和角度。
