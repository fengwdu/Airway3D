import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# chumpy still imports removed NumPy aliases on recent NumPy versions.
# Patch them before importing smplx/FLAME.
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.camera import backproject, coordinate_transform, merge_depth_frames
from core.capture import OrbbecCapture
from core.face_test import detect_face_bbox
from core.mask_detection import FACE_LABELS, FACE_NECK_LABELS, get_parser
from third_party.flame import FLAME, FlameConfig
from third_party.tufa.Test import get_lmk_single


DEPTH_WINDOW = 2
MIN_ALIGN_LANDMARKS = 8
EXCLUDED_ALIGN_LANDMARKS_68 = {17, 18, 19, 24, 25, 26}  # 1-based: 18,19,20,25,26,27


def capture_three_frames(cap):
    frames = []
    while len(frames) < 3:
        out = cap.read()
        if out is None:
            continue
        camera, rgb, depth = out
        frames.append((rgb.copy(), depth.copy(), camera))
    return frames


def save_mesh_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def save_points_obj(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in points:
            f.write(f"v {float(p[0]):.8f} {float(p[1]):.8f} {float(p[2]):.8f}\n")


def depth_window_values(depth, u, v):
    h, w = depth.shape
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    half_left = DEPTH_WINDOW // 2 - 1
    half_right = DEPTH_WINDOW - half_left

    x0 = max(0, ui - half_left)
    x1 = min(w, ui + half_right)
    y0 = max(0, vi - half_left)
    y1 = min(h, vi + half_right)
    vals = depth[y0:y1, x0:x1].astype(np.float32).reshape(-1)
    valid_vals = vals[vals > 0]
    mean_depth = float(valid_vals.mean()) if valid_vals.size > 0 else float("nan")
    return vals, int(valid_vals.size), mean_depth


def backproject_landmarks_2x2(uvs, depth, camera):
    points = np.full((len(uvs), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(uvs), dtype=bool)

    for i, (u, v) in enumerate(uvs):
        vals, _, _ = depth_window_values(depth, u, v)
        vals = vals[vals > 0]
        if vals.size == 0:
            continue

        z = float(vals.mean()) / 1000.0
        x = (float(u) - camera.cx) * z / camera.fx
        y = (float(v) - camera.cy) * z / camera.fy
        points[i] = [x, y, z]
        valid[i] = True

    points[valid] = coordinate_transform(points[valid])
    return points, valid


def save_landmark_depth_csv(path, landmarks_68, depth_frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    max_values = DEPTH_WINDOW * DEPTH_WINDOW
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["landmark", "frame", "u", "v", "valid_count", "mean_depth_mm"]
            + [f"depth_{i}_mm" for i in range(max_values)]
        )
        for lmk_id, (u, v) in enumerate(landmarks_68):
            for frame_id, depth in enumerate(depth_frames):
                vals, valid_count, mean_depth = depth_window_values(depth, u, v)
                vals_out = [int(x) for x in vals[:max_values]]
                vals_out += [""] * (max_values - len(vals_out))
                writer.writerow([
                    lmk_id,
                    frame_id,
                    float(u),
                    float(v),
                    valid_count,
                    mean_depth,
                    *vals_out,
                ])


def draw_landmarks_68(rgb, landmarks, bbox=None, valid_mask_51=None):
    vis = rgb.copy()
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for idx, (u, v) in enumerate(landmarks):
        pt = (int(round(float(u))), int(round(float(v))))
        if valid_mask_51 is not None and idx >= 17:
            ok = bool(valid_mask_51[idx - 17])
            color = (0, 255, 0) if ok else (255, 0, 0)
        else:
            color = (160, 160, 160)
        radius = 4 if 17 <= idx <= 47 else 3
        cv2.circle(vis, pt, radius, color, -1, lineType=cv2.LINE_AA)
        if 17 <= idx <= 47:
            cv2.circle(vis, pt, radius + 3, (255, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            str(idx),
            (pt[0] + 4, pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def face_neck_crop_xyxy(rgb, bbox):
    """YuNet bbox: extend downward to 2x height, then expand the resulting box by 1.2."""
    h_img, w_img = rgb.shape[:2]
    x, y, w, h = [float(v) for v in bbox]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + 2.0 * h

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    crop_w = (x2 - x1) * 1.2
    crop_h = (y2 - y1) * 1.2

    x1 = max(0, int(round(cx - crop_w / 2.0)))
    y1 = max(0, int(round(cy - crop_h / 2.0)))
    x2 = min(w_img, int(round(cx + crop_w / 2.0)))
    y2 = min(h_img, int(round(cy + crop_h / 2.0)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Invalid face+neck crop ROI.")
    return x1, y1, x2, y2


def parse_crop_mapped(rgb, bbox):
    x1, y1, x2, y2 = face_neck_crop_xyxy(rgb, bbox)
    crop = rgb[y1:y2, x1:x2]
    crop_parsing = get_parser().parse(crop)
    parsing = np.zeros(rgb.shape[:2], dtype=crop_parsing.dtype)
    parsing[y1:y2, x1:x2] = crop_parsing
    return parsing, (x1, y1, x2, y2)


def label_color(label):
    if label == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.array([
        (37 * int(label) + 53) % 256,
        (97 * int(label) + 29) % 256,
        (173 * int(label) + 71) % 256,
    ], dtype=np.uint8)


def save_all_label_overlay(path, rgb, parsing, bbox=None, crop_xyxy=None):
    color = np.zeros_like(rgb)
    for label in np.unique(parsing):
        if label == 0:
            continue
        color[parsing == label] = label_color(label)
    overlay = cv2.addWeighted(rgb, 0.55, color, 0.45, 0.0)
    overlay[parsing == 0] = rgb[parsing == 0]

    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def backproject_labels(parsing, depth, camera, labels):
    mask = np.isin(parsing, labels)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    uvs = np.stack([xs, ys], axis=1).astype(np.float32)
    pts_cam, _ = backproject(uvs, depth, camera)
    return coordinate_transform(pts_cam)


def save_face_parsing_preview(path_prefix, rgb, parsing, bbox, crop_xyxy):
    face_mask = np.isin(parsing, FACE_LABELS)
    face_neck_mask = np.isin(parsing, FACE_NECK_LABELS)
    neck_only_mask = face_neck_mask & ~face_mask

    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[face_mask] = (0, 255, 0)
    color[neck_only_mask] = (255, 180, 0)
    blended = cv2.addWeighted(overlay, 0.65, color, 0.35, 0.0)
    blended[~face_neck_mask] = rgb[~face_neck_mask]
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x1, y1, x2, y2 = crop_xyxy
    cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imwrite(str(path_prefix.with_name(path_prefix.name + "_overlay.jpg")),
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(path_prefix.with_name(path_prefix.name + "_mask.png")),
                (face_neck_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(path_prefix.with_name(path_prefix.name + "_labels.png")),
                parsing.astype(np.uint8))
    save_all_label_overlay(path_prefix.with_name(path_prefix.name + "_all_labels_overlay.jpg"),
                           rgb, parsing, bbox, crop_xyxy)

    labels, counts = np.unique(parsing, return_counts=True)
    with path_prefix.with_name(path_prefix.name + "_label_counts.csv").open("w", encoding="utf-8") as f:
        f.write("label,count\n")
        for label, count in zip(labels, counts):
            f.write(f"{int(label)},{int(count)}\n")


def sample_to_n(points, n, device):
    if points.shape[0] <= n:
        return points
    ids = torch.randperm(points.shape[0], device=device)[:n]
    return points[ids]


def rigid_umeyama(source, target):
    """Rigid source -> target alignment, row-vector convention: x @ R.T + t."""
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_c = source - source_mean
    target_c = target - target_mean

    h = target_c.transpose(0, 1) @ source_c
    u, _, vh = torch.linalg.svd(h)
    d = torch.eye(3, device=source.device, dtype=source.dtype)
    if torch.det(u @ vh) < 0:
        d[-1, -1] = -1
    r = u @ d @ vh
    t = target_mean.squeeze(0) - (r @ source_mean.squeeze(0))
    return r, t


def transform_points(points, r, t):
    return points @ r.transpose(0, 1) + t


def compute_vertex_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    return normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def nearest_vertex_ids(points, verts, batch_size=4096):
    chunks = []
    for start in range(0, points.shape[0], batch_size):
        chunk = points[start:start + batch_size]
        chunks.append(torch.cdist(chunk, verts).argmin(dim=1))
    return torch.cat(chunks, dim=0)


def flame_forward_transformed(flame, shape, expr, pose, trans, r, t):
    verts, lmks = flame(
        shape_params=shape,
        expression_params=expr,
        pose_params=pose,
        transl=trans,
    )
    return transform_points(verts[0], r, t), transform_points(lmks[0], r, t)


def code_style_loss(
    flame,
    shape,
    expr,
    pose,
    trans,
    points,
    point_vertex_ids,
    weights,
    r,
    t,
    vertex_normals_obs=None,
):
    verts_obs, _ = flame_forward_transformed(flame, shape, expr, pose, trans, r, t)
    matched = verts_obs[point_vertex_ids]
    diff = matched - points
    n_points = max(1, point_vertex_ids.numel())

    loss = verts_obs.new_zeros(())
    w_point = weights["point_cloud"]
    if w_point > 0:
        loss = loss + (w_point / n_points) * diff.pow(2).sum()

    w_plane = weights["point_plane"]
    if w_plane > 0 and vertex_normals_obs is not None:
        matched_normals = vertex_normals_obs[point_vertex_ids]
        proj = (matched_normals * diff).sum(dim=1)
        loss = loss + (w_plane / n_points) * proj.pow(2).sum()

    if weights["shape_reg"] > 0:
        loss = loss + weights["shape_reg"] * shape.pow(2).sum()
    if weights["expr_reg"] > 0:
        loss = loss + weights["expr_reg"] * expr.pow(2).sum()

    return 0.5 * loss


@torch.no_grad()
def loss_breakdown(flame, shape, expr, pose, trans, points, ids, weights, r, t, normals_obs):
    verts_obs, _ = flame_forward_transformed(flame, shape, expr, pose, trans, r, t)
    diff = verts_obs[ids] - points
    diff_sq = diff.pow(2).sum(dim=1)
    loss_pc = 0.5 * weights["point_cloud"] * diff_sq.mean()
    rms_pc = diff_sq.mean().sqrt()

    if normals_obs is not None and weights["point_plane"] > 0:
        proj = (normals_obs[ids] * diff).sum(dim=1)
        loss_pl = 0.5 * weights["point_plane"] * proj.pow(2).mean()
        rms_pl = proj.pow(2).mean().sqrt()
    else:
        loss_pl = verts_obs.new_zeros(())
        rms_pl = verts_obs.new_zeros(())
    loss_shape = 0.5 * weights["shape_reg"] * shape.pow(2).sum()
    loss_expr = 0.5 * weights["expr_reg"] * expr.pow(2).sum()

    return {
        "total": float((loss_pc + loss_pl + loss_shape + loss_expr).item()),
        "pc": float(loss_pc.item()),
        "pl": float(loss_pl.item()),
        "shape": float(loss_shape.item()),
        "expr": float(loss_expr.item()),
        "rms_pc": float(rms_pc.item()),
        "rms_pl": float(rms_pl.item()),
    }


def run_adam_stage(
    stage_name,
    flame,
    shape,
    expr,
    pose,
    trans,
    points,
    weights,
    r,
    t,
    optim_vars,
    args,
    lr,
    outer_iters,
    inner_iters,
    neck_first_only=False,
):
    optimizer = torch.optim.Adam(optim_vars, lr=lr)

    for outer in range(outer_iters):
        with torch.no_grad():
            verts_obs, _ = flame_forward_transformed(flame, shape, expr, pose, trans, r, t)
            ids = nearest_vertex_ids(points, verts_obs)
            normals_model = compute_vertex_normals(
                flame(
                    shape_params=shape,
                    expression_params=expr,
                    pose_params=pose,
                    transl=trans,
                )[0][0],
                flame.faces_tensor,
            )
            normals_obs = normals_model @ r.transpose(0, 1)

        prev_loss = None
        for step in range(inner_iters):
            optimizer.zero_grad()
            loss = code_style_loss(
                flame, shape, expr, pose, trans,
                points, ids, weights, r, t,
                vertex_normals_obs=normals_obs,
            )
            loss.backward()
            if neck_first_only and pose.grad is not None:
                pose.grad[:, 1:] = 0
            optimizer.step()

            loss_val = float(loss.item())
            if step % args.log_every == 0 or step == inner_iters - 1:
                print(
                    f"[{stage_name}] outer={outer:02d} step={step:04d} "
                    f"loss={loss_val:.6e}",
                    flush=True,
                )
            if prev_loss is not None and abs(prev_loss - loss_val) < args.tol_loss:
                break
            prev_loss = loss_val

        bd = loss_breakdown(flame, shape, expr, pose, trans, points, ids, weights, r, t, normals_obs)
        print(
            f"[{stage_name}] outer={outer:02d} end total={bd['total']:.6e} "
            f"pc={bd['pc']:.6e} pl={bd['pl']:.6e} "
            f"shape={bd['shape']:.6e} expr={bd['expr']:.6e} "
            f"rms_pc={bd['rms_pc']:.6e} rms_pl={bd['rms_pl']:.6e}",
            flush=True,
        )


def save_fit_state(path, flame, shape, expr, pose, trans, r, t):
    with torch.no_grad():
        verts, _ = flame_forward_transformed(flame, shape, expr, pose, trans, r, t)
    save_mesh_obj(path, verts.detach().cpu().numpy(), flame.faces_tensor.detach().cpu().numpy())


def prepare_capture(frames, out_dir, args):
    rgb = frames[0][0]
    camera = frames[0][2]
    depth = merge_depth_frames([frame[1] for frame in frames])

    bbox = detect_face_bbox(rgb)
    if bbox is None:
        raise RuntimeError("No YuNet face bbox detected.")

    lmk2d = np.asarray(get_lmk_single(rgb, bbox), dtype=np.float32)
    save_landmark_depth_csv(
        out_dir / "landmark_depths_3frames.csv",
        lmk2d,
        [frame[1] for frame in frames],
    )

    lmk3d_51, mask_51 = backproject_landmarks_2x2(lmk2d[17:], depth, camera)
    lmk_vis = draw_landmarks_68(rgb, lmk2d, bbox, mask_51)
    cv2.imwrite(str(out_dir / "capture_lmk68_depth_validity.jpg"), cv2.cvtColor(lmk_vis, cv2.COLOR_RGB2BGR))
    if mask_51[:31].sum() < MIN_ALIGN_LANDMARKS:
        raise RuntimeError(f"Not enough valid landmarks in 17-47: {int(mask_51[:31].sum())}/31")

    parsing, crop_xyxy = parse_crop_mapped(rgb, bbox)
    save_face_parsing_preview(out_dir / "face_parsing", rgb, parsing, bbox, crop_xyxy)

    pts_face = backproject_labels(parsing, depth, camera, FACE_LABELS)
    points = backproject_labels(parsing, depth, camera, FACE_NECK_LABELS)
    if pts_face.shape[0] == 0:
        raise RuntimeError("Face parsing failed.")
    if points.shape[0] == 0:
        raise RuntimeError("Empty face+neck point cloud.")

    vis = rgb.copy()
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x1, y1, x2, y2 = crop_xyxy
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.imwrite(str(out_dir / "capture_rgb.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return points.astype(np.float32), lmk3d_51.astype(np.float32), mask_51.astype(bool)


def fit_one(points_np, lmk3d_51_np, mask_51_np, out_dir, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32

    config = FlameConfig()
    config.batch_size = 1
    config.use_3D_translation = True
    flame = FLAME(config).to(device)
    flame.train()
    flame.head_pose = nn.Parameter(torch.zeros((1, 3), device=device, dtype=dtype), requires_grad=True)

    shape = torch.zeros((1, config.shape_params), device=device, dtype=dtype, requires_grad=True)
    expr = torch.zeros((1, config.expression_params), device=device, dtype=dtype, requires_grad=True)
    pose = torch.zeros((1, config.pose_params), device=device, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        pose[:, 0] = args.neck0_init
    trans = torch.zeros((1, 3), device=device, dtype=dtype, requires_grad=True)

    points = torch.as_tensor(points_np, device=device, dtype=dtype)
    points = points[torch.isfinite(points).all(dim=1)]
    points = sample_to_n(points, args.sample_points, device)

    lmk3d_51 = torch.as_tensor(lmk3d_51_np, device=device, dtype=dtype)
    mask_51 = torch.as_tensor(mask_51_np, device=device, dtype=torch.bool)

    with torch.no_grad():
        _, lmk0 = flame(
            shape_params=shape.detach(),
            expression_params=expr.detach(),
            pose_params=pose.detach(),
            transl=trans.detach(),
        )

        # FLAME static landmarks do not include the 68-point contour landmarks 0-16.
        # Therefore lmk0[:31] corresponds to original 68-point landmarks 17-47.
        align_mask = mask_51[:31].clone()
        for lmk_id in EXCLUDED_ALIGN_LANDMARKS_68:
            align_mask[lmk_id - 17] = False
        print(f"Excluded Umeyama landmarks (0-based): {sorted(EXCLUDED_ALIGN_LANDMARKS_68)}")
        model_align = lmk0[0, :31][align_mask]
        obs_align = lmk3d_51[:31][align_mask]
        finite_align = torch.isfinite(obs_align).all(dim=1)
        model_align = model_align[finite_align]
        obs_align = obs_align[finite_align]
        print(f"Finite Umeyama landmarks: {obs_align.shape[0]}/{int(align_mask.sum())}")
        if obs_align.shape[0] < 3:
            raise RuntimeError("Not enough finite landmarks for Umeyama alignment.")
        r, t = rigid_umeyama(model_align, obs_align)

        verts_init, _ = flame_forward_transformed(
            flame, shape.detach(), expr.detach(), pose.detach(), trans.detach(), r, t
        )
        save_mesh_obj(out_dir / "initial_flame_to_pointcloud.obj",
                      verts_init.detach().cpu().numpy(),
                      flame.faces_tensor.detach().cpu().numpy())
        save_points_obj(out_dir / "face_neck_point_cloud.obj", points.detach().cpu().numpy())
        save_points_obj(out_dir / "alignment_landmarks_17_47.obj", obs_align.detach().cpu().numpy())

    weights = {
        "point_cloud": args.point_weight,
        "point_plane": args.plane_weight,
        "shape_reg": args.shape_reg,
        "expr_reg": args.expr_reg,
    }
    run_adam_stage(
        "Stage1 head+neck0",
        flame, shape, expr, pose, trans, points, weights, r, t,
        optim_vars=[flame.head_pose, pose, trans],
        args=args,
        lr=args.stage1_lr,
        outer_iters=args.stage1_outer_iters,
        inner_iters=args.stage1_inner_iters,
        neck_first_only=True,
    )
    save_fit_state(out_dir / "stage1_flame_to_pointcloud.obj",
                   flame, shape, expr, pose, trans, r, t)
    torch.save(
        {
            "pose": pose.detach().cpu(),
            "trans": trans.detach().cpu(),
            "head_pose": flame.head_pose.detach().cpu(),
            "R_flame_to_pointcloud": r.detach().cpu(),
            "t_flame_to_pointcloud": t.detach().cpu(),
        },
        out_dir / "stage1_fit_result.pt",
    )

    run_adam_stage(
        "Stage2 all",
        flame, shape, expr, pose, trans, points, weights, r, t,
        optim_vars=[shape, expr, pose, trans, flame.head_pose],
        args=args,
        lr=args.lr,
        outer_iters=args.outer_iters,
        inner_iters=args.inner_iters,
    )

    with torch.no_grad():
        verts_final, _ = flame_forward_transformed(flame, shape, expr, pose, trans, r, t)
    save_mesh_obj(out_dir / "fitted_flame_to_pointcloud.obj",
                  verts_final.detach().cpu().numpy(),
                  flame.faces_tensor.detach().cpu().numpy())
    torch.save(
        {
            "shape": shape.detach().cpu(),
            "expr": expr.detach().cpu(),
            "pose": pose.detach().cpu(),
            "trans": trans.detach().cpu(),
            "head_pose": flame.head_pose.detach().cpu(),
            "R_flame_to_pointcloud": r.detach().cpu(),
            "t_flame_to_pointcloud": t.detach().cpu(),
        },
        out_dir / "fit_result.pt",
    )


def main():
    parser = argparse.ArgumentParser(description="OpenCV capture + FLAME-to-face-neck-pointcloud Adam fitting test.")
    parser.add_argument("--out-dir", default="tests/flame_fit_face_neck_output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-points", type=int, default=12000)
    parser.add_argument("--stage1-outer-iters", type=int, default=6)
    parser.add_argument("--stage1-inner-iters", type=int, default=120)
    parser.add_argument("--stage1-lr", type=float, default=5e-3)
    parser.add_argument("--neck0-init", type=float, default=-0.5)
    parser.add_argument("--outer-iters", type=int, default=12)
    parser.add_argument("--inner-iters", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tol-loss", type=float, default=1e-10)
    parser.add_argument("--point-weight", type=float, default=0.1)
    parser.add_argument("--plane-weight", type=float, default=0.9)
    parser.add_argument("--shape-reg", type=float, default=3e-8)
    parser.add_argument("--expr-reg", type=float, default=3e-8)
    parser.add_argument("--log-every", type=int, default=40)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = OrbbecCapture(timeout_ms=100)
    cap.start()
    try:
        print("Preview started. Press SPACE or C to capture and fit, Q or ESC to quit.")
        while True:
            out = cap.read()
            if out is None:
                continue
            _, rgb, _ = out
            cv2.imshow("FLAME Fit Preview", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return
            if key in (32, ord("c"), ord("C")):
                frames = capture_three_frames(cap)
                print("Captured 3 frames. Preparing face+neck point cloud...")
                try:
                    points_np, lmk3d_51_np, mask_51_np = prepare_capture(frames, out_dir, args)
                    print(f"Point cloud size before sampling: {points_np.shape[0]}")
                    print(f"Valid alignment landmarks 17-47: {int(mask_51_np[:31].sum())}/31")
                    fit_one(points_np, lmk3d_51_np, mask_51_np, out_dir, args)
                except Exception as exc:
                    print(f"Fit failed: {exc}")
                    continue
                print(f"Saved outputs to: {out_dir}")
    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
