import json
import numpy as np

class Camera:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def load_camera(camera_path = "./assets/camera.json"):
    """
    从 camera_path 读取相机内参并构建 Camera
    """
    with open(camera_path, "r", encoding="utf-8") as f:
        intr = json.load(f)

    return Camera(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])

def backproject(uvs, depth, camera):
    """
    最近邻深度查询的反投影, 适用于大量像素点(点云).
    uvs: (N, 2)
    """
    u = uvs[:, 0]
    v = uvs[:, 1]

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    H, W = depth.shape
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    z = np.zeros(len(uvs), dtype=np.float32)
    z[in_bounds] = depth[vi[in_bounds], ui[in_bounds]] / 1000  # (N, )
    valid = (z > 0) & in_bounds

    x = (u[valid] - camera.cx) * z[valid] / camera.fx
    y = (v[valid] - camera.cy) * z[valid] / camera.fy

    return np.stack([x, y, z[valid]], axis=1).astype(np.float32), valid


def backproject_bilinear(uvs, depth, camera):
    """
    双线性插值深度查询的反投影, 适用于亚像素精度的关键点.
    跳过深度为0的邻居, 只用有效像素加权, 减少边缘空洞导致的点丢失.
    uvs: (N, 2)
    """
    u = uvs[:, 0]
    v = uvs[:, 1]
    H, W = depth.shape

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    in_bounds = (u0 >= 0) & (u1 < W) & (v0 >= 0) & (v1 < H)

    wu = u - u0
    wv = v - v0

    z = np.zeros(len(uvs), dtype=np.float32)
    idx = np.where(in_bounds)[0]
    if len(idx) > 0:
        u0i, v0i, u1i, v1i = u0[idx], v0[idx], u1[idx], v1[idx]
        wu_i, wv_i = wu[idx], wv[idx]

        d00 = depth[v0i, u0i].astype(np.float32)
        d01 = depth[v0i, u1i].astype(np.float32)
        d10 = depth[v1i, u0i].astype(np.float32)
        d11 = depth[v1i, u1i].astype(np.float32)

        w00 = (1 - wu_i) * (1 - wv_i)
        w01 = wu_i       * (1 - wv_i)
        w10 = (1 - wu_i) * wv_i
        w11 = wu_i       * wv_i

        # 深度为0的邻居不参与插值
        m00 = (d00 > 0).astype(np.float32)
        m01 = (d01 > 0).astype(np.float32)
        m10 = (d10 > 0).astype(np.float32)
        m11 = (d11 > 0).astype(np.float32)

        w_sum = w00 * m00 + w01 * m01 + w10 * m10 + w11 * m11
        d_sum = w00 * m00 * d00 + w01 * m01 * d01 + w10 * m10 * d10 + w11 * m11 * d11

        good = w_sum > 0
        z[idx[good]] = d_sum[good] / w_sum[good] / 1000.0

    valid = (z > 0) & in_bounds

    x = (u[valid] - camera.cx) * z[valid] / camera.fx
    y = (v[valid] - camera.cy) * z[valid] / camera.fy

    return np.stack([x, y, z[valid]], axis=1).astype(np.float32), valid


def merge_depth_frames(depth_frames):
    """
    合并三帧深度图。0 视为无效深度，逐像素取非零中位数。
    """
    if len(depth_frames) == 0:
        raise ValueError("depth_frames is empty")

    stack = np.stack(depth_frames, axis=0).astype(np.float32)
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        merged = np.nanmedian(stack, axis=0)
    merged = np.nan_to_num(merged, nan=0.0)
    return np.clip(merged, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def backproject_window(uvs, depth, camera, window=4, trim=1, keep_shape=False):
    """
    对关键点使用邻域深度反投影。
    每个点取周围 window x window 个深度值，去掉两端极端值后求均值。
    keep_shape=True 时保留与 uvs 等长的输出，无效点填 NaN。
    """
    H, W = depth.shape
    half_left = window // 2 - 1
    half_right = window - half_left

    if keep_shape:
        points = np.full((len(uvs), 3), np.nan, dtype=np.float32)
    else:
        points = []
    valid = np.zeros(len(uvs), dtype=bool)

    for i, (u, v) in enumerate(uvs):
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        x0 = max(0, ui - half_left)
        x1 = min(W, ui + half_right)
        y0 = max(0, vi - half_left)
        y1 = min(H, vi + half_right)
        if x0 >= x1 or y0 >= y1:
            continue

        vals = depth[y0:y1, x0:x1].astype(np.float32).reshape(-1)
        vals = vals[vals > 0]
        if vals.size == 0:
            continue

        vals.sort()
        if trim > 0 and vals.size > 2 * trim:
            vals = vals[trim:-trim]

        z = float(vals.mean()) / 1000.0
        x = (float(u) - camera.cx) * z / camera.fx
        y = (float(v) - camera.cy) * z / camera.fy
        if keep_shape:
            points[i] = [x, y, z]
        else:
            points.append([x, y, z])
        valid[i] = True

    return np.asarray(points, dtype=np.float32), valid


def backprojection_inverse(points3d, camera):
    """
    points3d: [N, 3]
    """
    X, Y, Z = points3d[..., 0], points3d[..., 1], points3d[..., 2]
    x = camera.fx * X / Z + camera.cx                   
    y = camera.fy * Y / Z + camera.cy
    if hasattr(points3d, "detach"):
        import torch
        return torch.stack([x, y], dim=-1)
    return np.stack([x, y], axis=-1)

def save_2D(points2D, save_dir):
    with open(save_dir, 'w') as f:
        for pt in points2D:
            f.write(f'v {pt[0]:.8f} {pt[1]:.8f}\n')
    print(f'Save {len(points2D)} vertices to {save_dir}')


def save_3D(points3D, save_dir):
    with open(save_dir, 'w') as f:
        for pt in points3D:
            f.write(f'v {pt[0]:.8f} {-pt[1]:.8f} {-pt[2]:.8f}\n')           # 为了可视化方便，保存的实际上是(X, -Y, -Z)
    print(f'Save {len(points3D)} vertices to {save_dir}')

def coordinate_transform(points3D):
    """
    坐标系转换: 相机坐标系(Y朝下,Z朝前) → 模型坐标系(Y朝上,Z朝后)
    points3D: [N, 3]
    """
    out = points3D.copy()
    out[..., 1] = -out[..., 1]
    out[..., 2] = -out[..., 2]
    return out
