import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import trimesh

from core.camera import Camera

# chumpy still imports aliases removed from recent NumPy versions.
# Patch before importing FLAME/chumpy.
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

from third_party.flame import FLAME, FlameConfig


# 本模块在主流程中的职责：
# 汇总 landmark / 点云 / 相机参数，执行 FLAME 拟合优化，
# 产出闭口与张口两张最终网格（mesh_closed / mesh_open）。

ALIGN_LMK_RANGE_68 = (17, 48)
ALIGN_EXCLUDED_LMK_68 = {17, 18, 19, 24, 25, 26}
SEMANTIC_LMK_68 = [8, 21, 22, 33, 50, 51, 52, 56, 57, 58]
MOUTH_DISTANCE_PAIRS_68 = [(50, 58), (51, 57), (52, 56)]
NECK0_INIT = -0.5
POSE_ALLOWED_NECK0_JAW0 = (0, 3)


def load_2dpoints(keypoints_path):
    keypoints = []
    with open(keypoints_path, "r") as f:
        for line in f:
            if line.startswith("v"):
                part = line.strip().split()
                x, y = map(float, part[1:])
                keypoints.append([x, y])
    return torch.tensor(keypoints, dtype=torch.float32)


def load_3dpoints(ptcloud_path):
    ptcloud = []
    with open(ptcloud_path, "r") as f:
        for line in f:
            if line.startswith("v"):
                part = line.strip().split()
                x, y, z = map(float, part[1:])
                ptcloud.append([x, y, z])
    return torch.tensor(ptcloud, dtype=torch.float32)


def sample_to_N(pc, N):
    if pc.shape[0] <= N:
        return pc
    idx = torch.randperm(pc.shape[0], device=pc.device)[:N]
    return pc[idx]


def _export_points_obj(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        for p in points:
            if np.isfinite(p).all():
                f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def _export_open_mouth_lmk_debug(out_dir, gt_open, pred_open):
    os.makedirs(out_dir, exist_ok=True)
    _export_points_obj(os.path.join(out_dir, "lmk_open_gt.obj"), gt_open)
    _export_points_obj(os.path.join(out_dir, "lmk_open_flame_pred.obj"), pred_open)

    gt_dist = float(np.linalg.norm(gt_open[51] - gt_open[57]))
    pred_dist = float(np.linalg.norm(pred_open[51] - pred_open[57]))
    with open(os.path.join(out_dir, "mouth_lmk_distances.txt"), "w", encoding="utf-8") as f:
        f.write("landmark_pair: 51-57\n")
        f.write(f"gt_open_distance_m: {gt_dist:.8f}\n")
        f.write(f"gt_open_distance_cm: {gt_dist * 100.0:.4f}\n")
        f.write(f"flame_open_distance_m: {pred_dist:.8f}\n")
        f.write(f"flame_open_distance_cm: {pred_dist * 100.0:.4f}\n")


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _rigid_umeyama(source, target):
    """Rigid source -> target alignment, row-vector convention: source @ R.T + t."""
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_c = source - source_mean
    target_c = target - target_mean

    H = target_c.transpose(0, 1) @ source_c
    U, _, Vh = torch.linalg.svd(H)
    D = torch.eye(3, device=source.device, dtype=source.dtype)
    if torch.det(U @ Vh) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vh
    t = target_mean.squeeze(0) - (R @ source_mean.squeeze(0))
    return R, t


def _transform_points(points, R, t):
    return points @ R.transpose(0, 1) + t


def _pick_alignment_points(model_points, gt_points, mask):
    valid_index = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if gt_points.shape[0] == mask.numel():
        gt_valid = gt_points[valid_index]
    elif gt_points.shape[0] == valid_index.numel():
        gt_valid = gt_points
    else:
        n = min(valid_index.numel(), gt_points.shape[0])
        valid_index = valid_index[:n]
        gt_valid = gt_points[:n]

    align_start, align_stop = ALIGN_LMK_RANGE_68
    keep = (valid_index >= align_start) & (valid_index < align_stop)
    for excluded in ALIGN_EXCLUDED_LMK_68:
        keep = keep & (valid_index != excluded)
    keep = keep & torch.isfinite(gt_valid).all(dim=1)
    if keep.sum() < 3:
        keep = ((valid_index >= align_start) & (valid_index < align_stop)
                & torch.isfinite(gt_valid).all(dim=1))

    model_pts = model_points[valid_index[keep]]
    gt_pts = gt_valid[keep]
    if model_pts.shape[0] < 3:
        raise ValueError("有效 3D landmark 少于 3 个，无法做 FLAME 到观测空间的刚性对齐")
    return model_pts, gt_pts


def _flame_to_observation_transform(model_lmk, gt_lmk, mask):
    model_pts, gt_pts = _pick_alignment_points(model_lmk, gt_lmk, mask)
    return _rigid_umeyama(model_pts, gt_pts)


def _transform_vertices_batch(verts, rotations, translations):
    return torch.stack([
        _transform_points(verts[b], rotations[b], translations[b])
        for b in range(verts.shape[0])
    ], dim=0)


def _forward_observation_space_with_landmarks(flame, shape_params, expr_params, pose_params,
                                              trans_params, rotations, translations):
    verts, landmarks = flame(
        shape_params=shape_params,
        expression_params=expr_params,
        pose_params=pose_params,
        transl=trans_params,
    )
    return (
        _transform_vertices_batch(verts, rotations, translations),
        _transform_vertices_batch(landmarks, rotations, translations),
    )


def _compute_vertex_normals_batch(verts, faces):
    normals = []
    for b in range(verts.shape[0]):
        vb = verts[b]
        v0 = vb[faces[:, 0]]
        v1 = vb[faces[:, 1]]
        v2 = vb[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        vertex_normals = torch.zeros_like(vb)
        vertex_normals.index_add_(0, faces[:, 0], face_normals)
        vertex_normals.index_add_(0, faces[:, 1], face_normals)
        vertex_normals.index_add_(0, faces[:, 2], face_normals)
        normals.append(vertex_normals / vertex_normals.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return torch.stack(normals, dim=0)


@torch.no_grad()
def _nearest_vertex_ids(points, verts, batch_size=4096):
    chunks = []
    for start in range(0, points.shape[0], batch_size):
        chunk = points[start:start + batch_size]
        chunks.append(torch.cdist(chunk, verts).argmin(dim=1))
    return torch.cat(chunks, dim=0)


def _forward_observation_space(flame, shape_params, expr_params, pose_params,
                               trans_params, rotations, translations):
    verts, _ = _forward_observation_space_with_landmarks(
        flame, shape_params, expr_params, pose_params, trans_params,
        rotations, translations,
    )
    return verts


@torch.no_grad()
def _update_correspondences(flame, shape_params, expr_params, pose_params,
                            trans_params, ptcloud_list, rotations, translations):
    verts = _forward_observation_space(
        flame, shape_params, expr_params, pose_params, trans_params,
        rotations, translations,
    )
    return [_nearest_vertex_ids(points, verts[b]) for b, points in enumerate(ptcloud_list)]


@torch.no_grad()
def _frozen_vertex_normals(flame, shape_params, expr_params, pose_params,
                           trans_params, rotations):
    verts, _ = flame(
        shape_params=shape_params,
        expression_params=expr_params,
        pose_params=pose_params,
        transl=trans_params,
    )
    normals = _compute_vertex_normals_batch(verts, flame.faces_tensor)
    return torch.stack([
        normals[b] @ rotations[b].transpose(0, 1)
        for b in range(normals.shape[0])
    ], dim=0)


def _landmark_constraint_loss(landmarks, lmk_gt_list, lmk_mask_list,
                              weights, use_mouth_distance, use_landmark_points):
    loss = landmarks.new_zeros(())

    if use_mouth_distance and weights.get("mouth_distance", 0.0) > 0:
        # 闭口帧把三对内唇距离压向 0；张口帧跟随真实 3D 内唇距离。
        mouth_losses = []
        for b, (gt, mask) in enumerate(zip(lmk_gt_list, lmk_mask_list)):
            for i, j in MOUTH_DISTANCE_PAIRS_68:
                if bool(mask[i]) and bool(mask[j]):
                    pred = torch.linalg.norm(landmarks[b, i] - landmarks[b, j])
                    if b == 0:
                        target = pred.new_zeros(())
                    else:
                        target = torch.linalg.norm(gt[i] - gt[j])
                    if torch.isfinite(target):
                        mouth_losses.append((pred - target).pow(2))
        if mouth_losses:
            loss = loss + weights["mouth_distance"] * torch.stack(mouth_losses).mean()

    if use_landmark_points and weights.get("landmark_points", 0.0) > 0:
        for b, (gt, mask) in enumerate(zip(lmk_gt_list, lmk_mask_list)):
            idx = torch.tensor(SEMANTIC_LMK_68, device=landmarks.device, dtype=torch.long)
            valid = mask[idx] & torch.isfinite(gt[idx]).all(dim=1)
            if bool(valid.any()):
                pred = landmarks[b, idx[valid]]
                target = gt[idx[valid]]
                loss = loss + weights["landmark_points"] * (pred - target).pow(2).mean()

    return loss


def _mask_pose_grad(pose_params, allowed_indices):
    if pose_params.grad is None:
        return
    mask = torch.zeros_like(pose_params.grad)
    mask[:, list(allowed_indices)] = 1
    pose_params.grad.mul_(mask)


def _code_style_loss(
    flame,
    shape_params,
    expr_params,
    pose_params,
    trans_params,
    ptcloud_list,
    pc_vertex_ids,
    weights,
    rotations,
    translations,
    vertex_normals=None,
    lmk_gt_list=None,
    lmk_mask_list=None,
    use_mouth_distance=False,
    use_landmark_points=False,
):
    """
    Loss aligned with 代码/losses.py:
      0.5 * (point-cloud mean + optional point-plane mean + parameter regularizers).
    """
    verts, landmarks = _forward_observation_space_with_landmarks(
        flame, shape_params, expr_params, pose_params, trans_params,
        rotations, translations,
    )
    loss = verts.new_zeros(())

    for b, points in enumerate(ptcloud_list):
        ids = pc_vertex_ids[b]
        matched = verts[b, ids]
        diff = matched - points
        n_points = max(1, ids.numel())

        if weights["point_cloud"] > 0:
            loss = loss + (weights["point_cloud"] / n_points) * diff.pow(2).sum()

        if weights.get("point_plane", 0.0) > 0 and vertex_normals is not None:
            matched_normals = vertex_normals[b, ids]
            proj = (matched_normals * diff).sum(dim=1)
            loss = loss + (weights["point_plane"] / n_points) * proj.pow(2).sum()

    if weights["shape_reg"] > 0:
        loss = loss + weights["shape_reg"] * shape_params.pow(2).sum()
    if weights["expr_reg"] > 0:
        loss = loss + weights["expr_reg"] * expr_params.pow(2).sum()
    if lmk_gt_list is not None and lmk_mask_list is not None:
        loss = loss + _landmark_constraint_loss(
            landmarks,
            lmk_gt_list,
            lmk_mask_list,
            weights,
            use_mouth_distance=use_mouth_distance,
            use_landmark_points=use_landmark_points,
        )
    return 0.5 * loss


@torch.no_grad()
def _code_style_breakdown(
    flame,
    shape_params,
    expr_params,
    pose_params,
    trans_params,
    ptcloud_list,
    pc_vertex_ids,
    weights,
    rotations,
    translations,
    vertex_normals=None,
    lmk_gt_list=None,
    lmk_mask_list=None,
    use_mouth_distance=False,
    use_landmark_points=False,
):
    verts, landmarks = _forward_observation_space_with_landmarks(
        flame, shape_params, expr_params, pose_params, trans_params,
        rotations, translations,
    )
    loss_pc = verts.new_zeros(())
    loss_pl = verts.new_zeros(())
    rms_pc_acc = verts.new_zeros(())
    rms_pl_acc = verts.new_zeros(())

    for b, points in enumerate(ptcloud_list):
        ids = pc_vertex_ids[b]
        diff = verts[b, ids] - points
        diff_sq = diff.pow(2).sum(dim=1)
        loss_pc = loss_pc + 0.5 * weights["point_cloud"] * diff_sq.mean()
        rms_pc_acc = rms_pc_acc + diff_sq.mean()

        if weights.get("point_plane", 0.0) > 0 and vertex_normals is not None:
            proj = (vertex_normals[b, ids] * diff).sum(dim=1)
            loss_pl = loss_pl + 0.5 * weights["point_plane"] * proj.pow(2).mean()
            rms_pl_acc = rms_pl_acc + proj.pow(2).mean()

    n_frames = max(1, len(ptcloud_list))
    loss_shape = 0.5 * weights["shape_reg"] * shape_params.pow(2).sum()
    loss_expr = 0.5 * weights["expr_reg"] * expr_params.pow(2).sum()
    loss_lmk = verts.new_zeros(())
    if lmk_gt_list is not None and lmk_mask_list is not None:
        loss_lmk = 0.5 * _landmark_constraint_loss(
            landmarks,
            lmk_gt_list,
            lmk_mask_list,
            weights,
            use_mouth_distance=use_mouth_distance,
            use_landmark_points=use_landmark_points,
        )
    total = loss_pc + loss_pl + loss_shape + loss_expr + loss_lmk
    return {
        "loss_total": float(total.item()),
        "loss_pc": float(loss_pc.item()),
        "loss_pl": float(loss_pl.item()),
        "loss_lmk": float(loss_lmk.item()),
        "loss_shape": float(loss_shape.item()),
        "loss_expr": float(loss_expr.item()),
        "rms_pc": float((rms_pc_acc / n_frames).sqrt().item()),
        "rms_pl": float((rms_pl_acc / n_frames).sqrt().item()),
    }


def _run_adam_stage(stage_name, flame, shape_params, expr_params, pose_params,
                    trans_params, ptcloud_fit, rotations, translations, weights,
                    optimizer, outer_iters, inner_iters, tol_loss,
                    lmk_gt_list=None, lmk_mask_list=None,
                    use_mouth_distance=False, use_landmark_points=False,
                    pose_allowed_indices=None):
    for outer in range(outer_iters):
        pc_vertex_ids = _update_correspondences(
            flame, shape_params, expr_params, pose_params, trans_params,
            ptcloud_fit, rotations, translations,
        )
        vertex_normals = _frozen_vertex_normals(
            flame, shape_params, expr_params, pose_params, trans_params, rotations,
        )

        prev_loss = None
        for step in range(inner_iters):
            optimizer.zero_grad()
            loss = _code_style_loss(
                flame,
                shape_params,
                expr_params,
                pose_params,
                trans_params,
                ptcloud_fit,
                pc_vertex_ids,
                weights,
                rotations,
                translations,
                vertex_normals=vertex_normals,
                lmk_gt_list=lmk_gt_list,
                lmk_mask_list=lmk_mask_list,
                use_mouth_distance=use_mouth_distance,
                use_landmark_points=use_landmark_points,
            )
            loss.backward()
            if pose_allowed_indices is not None:
                _mask_pose_grad(pose_params, pose_allowed_indices)
            optimizer.step()

            loss_val = float(loss.item())
            if step % 30 == 0:
                print(f"[{stage_name}] outer={outer:02d} step={step:03d} loss={loss_val:.6e}")
            if prev_loss is not None and abs(prev_loss - loss_val) < tol_loss:
                break
            prev_loss = loss_val

        bd = _code_style_breakdown(
            flame,
            shape_params,
            expr_params,
            pose_params,
            trans_params,
            ptcloud_fit,
            pc_vertex_ids,
            weights,
            rotations,
            translations,
            vertex_normals=vertex_normals,
            lmk_gt_list=lmk_gt_list,
            lmk_mask_list=lmk_mask_list,
            use_mouth_distance=use_mouth_distance,
            use_landmark_points=use_landmark_points,
        )
        print(
            f"[{stage_name}] outer={outer:02d} end total={bd['loss_total']:.6e} "
            f"pc={bd['loss_pc']:.6e} pl={bd['loss_pl']:.6e} "
            f"lmk={bd['loss_lmk']:.6e} rms_pc={bd['rms_pc']:.6e}"
        )


def train(
    lmk3d_gt_closed,
    lmk3d_gt_open,
    lmk2d_gt_closed,
    lmk2d_gt_open,
    lmk_mask_closed,
    lmk_mask_open,
    ptcloud_closed,
    ptcloud_open,
    camera,
    rgb_closed=None,
    rgb_open=None,
    ptcloud_face_closed=None,
    ptcloud_face_open=None,
    device="cuda:0",
):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[FLAME] CUDA 不可用，自动回退到 CPU")
        device = "cpu"

    config = FlameConfig()
    config.use_3D_translation = True
    config.use_face_contour = True
    flame = FLAME(config).to(device)
    flame.train()

    lmk3d_gt_closed = torch.as_tensor(lmk3d_gt_closed, dtype=torch.float32, device=device)
    lmk3d_gt_open = torch.as_tensor(lmk3d_gt_open, dtype=torch.float32, device=device)
    lmk_mask_closed = torch.as_tensor(lmk_mask_closed, dtype=torch.bool, device=device)
    lmk_mask_open = torch.as_tensor(lmk_mask_open, dtype=torch.bool, device=device)
    if lmk3d_gt_closed.shape[0] < 68 or lmk3d_gt_open.shape[0] < 68:
        raise ValueError("当前主流程需要完整 68 点 landmark，以匹配 FLAME 动态轮廓点")
    lmk_gt_list = [lmk3d_gt_closed, lmk3d_gt_open]
    lmk_mask_list = [lmk_mask_closed, lmk_mask_open]
    ptcloud_closed_t = torch.as_tensor(ptcloud_closed, dtype=torch.float32, device=device)
    ptcloud_open_t = torch.as_tensor(ptcloud_open, dtype=torch.float32, device=device)
    ptcloud_face_closed_t = (
        torch.as_tensor(ptcloud_face_closed, dtype=torch.float32, device=device)
        if ptcloud_face_closed is not None else None
    )
    ptcloud_face_open_t = (
        torch.as_tensor(ptcloud_face_open, dtype=torch.float32, device=device)
        if ptcloud_face_open is not None else None
    )

    shape_params = torch.zeros((2, config.shape_params), device=device)
    expr_params = torch.zeros((2, config.expression_params), device=device)
    pose_params = torch.zeros((2, config.pose_params), device=device)
    pose_params[:, 0] = NECK0_INIT
    head_pose = torch.zeros((2, 3), device=device)

    trans_params = torch.zeros((2, 3), device=device)
    flame.head_pose = nn.Parameter(head_pose.clone(), requires_grad=True)

    with torch.no_grad():
        _, init_lmk = flame(
            shape_params=shape_params,
            expression_params=expr_params,
            pose_params=pose_params,
            transl=trans_params,
        )

    # 用 landmark 估计一次刚性变换，把 FLAME 放到观测点云空间。
    R_c, t_c = _flame_to_observation_transform(
        init_lmk[0],
        lmk3d_gt_closed,
        lmk_mask_closed,
    )
    R_o, t_o = _flame_to_observation_transform(
        init_lmk[1],
        lmk3d_gt_open,
        lmk_mask_open,
    )
    rotations = torch.stack([R_c, R_o], dim=0)
    translations = torch.stack([t_c, t_o], dim=0)

    if ptcloud_closed_t.shape[0] == 0 or ptcloud_open_t.shape[0] == 0:
        raise ValueError("点云为空，无法进行 FLAME 拟合")

    fit_closed = ptcloud_closed_t
    fit_open = ptcloud_open_t

    sample_ratio = 0.35
    sample_cap = 8000
    n_fit = max(
        1,
        min(
            int(fit_closed.shape[0] * sample_ratio),
            int(fit_open.shape[0] * sample_ratio),
            sample_cap,
        ),
    )
    ptcloud_fit = [sample_to_N(fit_closed, n_fit), sample_to_N(fit_open, n_fit)]

    shape_params = shape_params.detach().clone().requires_grad_(True)
    expr_params = expr_params.detach().clone().requires_grad_(True)
    pose_params = pose_params.detach().clone().requires_grad_(True)
    trans_params = trans_params.detach().clone().requires_grad_(True)

    weights = dict(
        point_cloud=0.1,
        point_plane=0.9,
        mouth_distance=10.0,
        landmark_points=2.0,
        shape_reg=1e-6,
        expr_reg=1e-6,
    )

    _sync_if_cuda(device)
    start_time = time.time()
    tol_loss = 1e-10

    stage1_optimizer = torch.optim.Adam([pose_params, flame.head_pose, trans_params], lr=5e-3)
    _run_adam_stage(
        "Stage1 head+neck0",
        flame, shape_params, expr_params, pose_params, trans_params,
        ptcloud_fit, rotations, translations, weights,
        stage1_optimizer,
        outer_iters=6,
        inner_iters=120,
        tol_loss=tol_loss,
        lmk_gt_list=lmk_gt_list,
        lmk_mask_list=lmk_mask_list,
        use_mouth_distance=True,
        use_landmark_points=False,
        pose_allowed_indices=POSE_ALLOWED_NECK0_JAW0,
    )

    stage2_optimizer = torch.optim.Adam(
        [shape_params, expr_params, pose_params, flame.head_pose, trans_params],
        lr=1e-2,
    )
    _run_adam_stage(
        "Stage2 all",
        flame, shape_params, expr_params, pose_params, trans_params,
        ptcloud_fit, rotations, translations, weights,
        stage2_optimizer,
        outer_iters=12,
        inner_iters=160,
        tol_loss=tol_loss,
        lmk_gt_list=lmk_gt_list,
        lmk_mask_list=lmk_mask_list,
        use_mouth_distance=True,
        use_landmark_points=True,
        pose_allowed_indices=POSE_ALLOWED_NECK0_JAW0,
    )

    _sync_if_cuda(device)
    print(f"[Timing] FLAME Adam fitting: {time.time() - start_time:.2f}s")

    with torch.no_grad():
        verts_model, lmk_model = flame(
            shape_params=shape_params.detach(),
            expression_params=expr_params.detach(),
            pose_params=pose_params.detach(),
            transl=trans_params.detach(),
        )
        verts_final = _transform_vertices_batch(verts_model, rotations, translations)
        lmk_final = _transform_vertices_batch(lmk_model, rotations, translations)
        faces_np = flame.faces_tensor.detach().cpu().numpy()
        mesh_closed = trimesh.Trimesh(
            vertices=verts_final[0].cpu().numpy(), faces=faces_np, process=False
        )
        mesh_open = trimesh.Trimesh(
            vertices=verts_final[1].cpu().numpy(), faces=faces_np, process=False
        )
        _export_open_mouth_lmk_debug(
            "./output/open",
            lmk3d_gt_open.detach().cpu().numpy(),
            lmk_final[1].detach().cpu().numpy(),
        )

    return mesh_closed, mesh_open


if __name__ == "__main__":
    camera_name = "camera"
    camera_intrinsic_path = os.path.join("assets", camera_name + ".json")

    with open(camera_intrinsic_path, "r") as f:
        intrinst_data = json.load(f)

    camera = Camera(
        fx=intrinst_data["fx"], fy=intrinst_data["fy"],
        cx=intrinst_data["cx"], cy=intrinst_data["cy"],
    )

    base_dir = os.path.join("result", camera_name, "landmark")
    save_dir = os.path.join("result", camera_name, "result")
    os.makedirs(save_dir, exist_ok=True)

    closed_dir = os.path.join(base_dir, "0")
    lmk3d_gt_closed = load_3dpoints(os.path.join(closed_dir, "lmk3d_0.obj"))
    ptcloud_closed = load_3dpoints(os.path.join(closed_dir, "mask_0.obj"))
    lmk_mask_closed = np.load(os.path.join(closed_dir, "avail_0.npy"))
    lmk_gt_closed = load_2dpoints(os.path.join(closed_dir, "lmk2d_0.txt"))

    open_dir = os.path.join(base_dir, "1")
    lmk3d_gt_open = load_3dpoints(os.path.join(open_dir, "lmk3d_1.obj"))
    ptcloud_open = load_3dpoints(os.path.join(open_dir, "mask_1.obj"))
    lmk_mask_open = np.load(os.path.join(open_dir, "avail_1.npy"))
    lmk_gt_open = load_2dpoints(os.path.join(open_dir, "lmk2d_1.txt"))

    mesh_closed, mesh_open = train(
        lmk3d_gt_closed=lmk3d_gt_closed,
        lmk3d_gt_open=lmk3d_gt_open,
        lmk2d_gt_closed=lmk_gt_closed,
        lmk2d_gt_open=lmk_gt_open,
        lmk_mask_closed=lmk_mask_closed,
        lmk_mask_open=lmk_mask_open,
        ptcloud_closed=ptcloud_closed,
        ptcloud_open=ptcloud_open,
        camera=camera,
    )

    out_closed = os.path.join(save_dir, "0")
    out_open = os.path.join(save_dir, "1")
    os.makedirs(out_closed, exist_ok=True)
    os.makedirs(out_open, exist_ok=True)
    mesh_closed.export(os.path.join(out_closed, "mesh_closed.obj"))
    mesh_open.export(os.path.join(out_open, "mesh_open.obj"))
    print("[OK] closed / open meshes saved")
