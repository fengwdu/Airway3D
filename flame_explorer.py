"""
FLAME Parameter Explorer — 交互式参数可视化
============================================
拖动滑块实时观察 shape / expression / pose 对人头网格的影响。
鼠标左键旋转 | 滚轮缩放 | 中键平移

用法:   python flame_explorer.py
依赖:   pip install pyvista
模型:   需要将 flame2023.pkl 放到 assets/ 目录
        (从 https://flame.is.tue.mpg.de/ 下载)
"""

import sys
import os
import types
import numpy as np

# ── chumpy stub: 让 pickle 能反序列化 FLAME pkl 中的 chumpy 对象 ──
if "chumpy" not in sys.modules:
    _chumpy = types.ModuleType("chumpy")
    _chumpy_ch = types.ModuleType("chumpy.ch")

    class _Ch(np.ndarray):
        """numpy ndarray 子类，仅用于 pickle 反序列化。"""
        def __new__(cls, *args, **kwargs):
            if not args and not kwargs:
                return np.ndarray.__new__(cls, shape=(0,), dtype=np.float64)
            if args and isinstance(args[0], (np.ndarray, list, tuple)):
                return np.asarray(args[0]).view(cls)
            return np.ndarray.__new__(cls, *args, **kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                arr = state.get("x", state.get("_data", np.array([])))
                super().__setstate__(np.asarray(arr).__reduce__()[2])
            else:
                super().__setstate__(state)

    _chumpy.Ch = _Ch
    _chumpy.ch = _chumpy_ch
    _chumpy_ch.Ch = _Ch
    for _sub in ("utils", "reordering", "logic"):
        _m = types.ModuleType(f"chumpy.{_sub}")
        setattr(_chumpy, _sub, _m)
        sys.modules[f"chumpy.{_sub}"] = _m
    sys.modules["chumpy"] = _chumpy
    sys.modules["chumpy.ch"] = _chumpy_ch

import torch
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from third_party.flame import FLAME, FlameConfig

# ───────────────────────── FLAME 初始化 ─────────────────────────

config = FlameConfig(
    batch_size=1,
    shape_params=300,
    expression_params=100,
    pose_params=6,
    use_face_contour=False,
    use_3D_translation=False,
)
flame = FLAME(config)
flame.eval()

faces_np = flame.faces_tensor.numpy()
pv_faces = np.column_stack([np.full(len(faces_np), 3), faces_np]).ravel()

# ───────────────────────── 参数状态 ─────────────────────────

state = dict(
    shape=np.zeros(300, dtype=np.float32),
    expr=np.zeros(100, dtype=np.float32),
    head=np.zeros(3, dtype=np.float32),
    neck=np.zeros(3, dtype=np.float32),
    jaw=np.zeros(3, dtype=np.float32),
    leye=np.zeros(3, dtype=np.float32),
    reye=np.zeros(3, dtype=np.float32),
)


def compute_mesh():
    with torch.no_grad():
        flame.head_pose.data = torch.from_numpy(state["head"]).unsqueeze(0)
        verts, lmks = flame(
            shape_params=torch.from_numpy(state["shape"]).unsqueeze(0),
            expression_params=torch.from_numpy(state["expr"]).unsqueeze(0),
            pose_params=torch.from_numpy(
                np.concatenate([state["neck"], state["jaw"]])
            ).unsqueeze(0),
            eye_pose=torch.from_numpy(
                np.concatenate([state["leye"], state["reye"]])
            ).unsqueeze(0),
        )
    return verts[0].numpy(), lmks[0].numpy()


init_verts, init_lmks = compute_mesh()
mesh = pv.PolyData(init_verts, pv_faces)
lmk_cloud = pv.PolyData(init_lmks)


def refresh(_=None):
    v, l = compute_mesh()
    mesh.points = v
    lmk_cloud.points = l


# ───────────────────────── 回调工厂 ─────────────────────────

def _cb_shape(idx):
    def cb(val):
        state["shape"][idx] = val
        refresh()
    return cb


def _cb_expr(idx):
    def cb(val):
        state["expr"][idx] = val
        refresh()
    return cb


def _cb_pose(group, axis):
    def cb(val):
        state[group][axis] = val
        refresh()
    return cb


# ───────────────────────── 构建 UI ─────────────────────────

pl = pv.Plotter(title="FLAME Parameter Explorer", window_size=(1400, 900))

pl.add_mesh(
    mesh, color="peachpuff", smooth_shading=True, pbr=True,
    metallic=0.0, roughness=0.55, name="face",
)
pl.add_mesh(
    lmk_cloud, color="red", point_size=6,
    render_points_as_spheres=True, name="lmk",
)

# ---------- Shape 滑块 (左侧 5 个) ----------
for i in range(5):
    pl.add_slider_widget(
        _cb_shape(i),
        rng=(-3.0, 3.0), value=0.0,
        title=f"Shape {i}",
        pointa=(0.01, 0.91 - i * 0.075),
        pointb=(0.21, 0.91 - i * 0.075),
        style="modern",
    )

# ---------- Expression 滑块 (右侧 5 个) ----------
for i in range(5):
    pl.add_slider_widget(
        _cb_expr(i),
        rng=(-3.0, 3.0), value=0.0,
        title=f"Expr {i}",
        pointa=(0.79, 0.91 - i * 0.075),
        pointb=(0.99, 0.91 - i * 0.075),
        style="modern",
    )

# ---------- Pose 滑块 (底部) ----------
pose_defs = [
    ("Head Yaw",   "head", 1, (-1.0, 1.0)),
    ("Head Pitch", "head", 0, (-0.8, 0.8)),
    ("Head Roll",  "head", 2, (-0.6, 0.6)),
    ("Jaw Pitch",  "jaw",  0, (-0.2, 0.8)),
    ("Jaw Slide",  "jaw",  1, (-0.5, 0.5)),
    ("Jaw Tilt",   "jaw",  2, (-0.3, 0.3)),
    ("Neck Bend",  "neck", 0, (-0.5, 0.5)),
]
n_pose = len(pose_defs)
sw = 0.12
gap = (0.94 - n_pose * sw) / max(n_pose - 1, 1)
for i, (title, grp, axis, rng) in enumerate(pose_defs):
    x0 = 0.03 + i * (sw + gap)
    pl.add_slider_widget(
        _cb_pose(grp, axis),
        rng=rng, value=0.0,
        title=title,
        pointa=(x0, 0.06),
        pointb=(x0 + sw, 0.06),
        style="modern",
    )

# ---------- 视角与提示 ----------
pl.view_xy()
pl.camera.zoom(1.3)
pl.add_text(
    "FLAME Explorer  |  Left-drag: rotate  Scroll: zoom  Mid-drag: pan",
    position="upper_edge", font_size=10, color="gray",
)
pl.show()
