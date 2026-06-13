import numpy as np
import trimesh
import json
import math

# 本模块在主流程中的职责：
# 基于拟合完成的闭口/张口网格，计算展示给用户的临床评估指标。

IDX = {
    "tragus_L": 1203,
    "tragus_R": 365,

    "earlobe_L": 1883,
    "earlobe_R": 544,

    "neck_ring": [
        3494, 1360, 2531, 1213, 1212, 2976, 3184, 3185,
        222, 221, 2964, 3012, 2151, 1948, 1886, 14,
        15, 3187, 3186, 1898, 374, 375, 1386, 463
    ]
}

LMK = {
    "chin": 8,
    "brow_L": 21,
    "brow_R": 22,
    "nose_tip": 33,
    "upper_lip": 51,
    "lower_lip": 57,
}

def angle_between(v1, v2):
    """计算两个向量夹角（度）"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def point_to_line_distance(P, A, B):
    """
    点 P 到直线 AB 的距离
    """
    AB = B - A
    AP = P - A
    return np.linalg.norm(np.cross(AP, AB)) / np.linalg.norm(AB)

def rodrigues_rotate(point, center, axis_unit, theta_rad):
    """
    点 point 绕过 center、沿 axis_unit 旋转 theta_rad (Rodrigues 公式)
    """
    p = point - center
    k = axis_unit
    p_rot = (
        p * np.cos(theta_rad)
        + np.cross(k, p) * np.sin(theta_rad)
        + k * np.dot(k, p) * (1 - np.cos(theta_rad))
    )
    return p_rot + center

def metrics_calculate(mesh_closed, mesh_open, adam_3d, lmk3d_closed, lmk3d_open):
    """
    adam_3d: (3,) 模型坐标系下的喉结 3D 坐标，来自强制 ArUco 检测。
    lmk3d_closed/open: (68, 3) 模型坐标系下的 TUFA landmark 反投影结果。
    """

    # 读取数据
    V_c = np.asarray(mesh_closed.vertices)
    V_o = np.asarray(mesh_open.vertices)
    L_c = np.asarray(lmk3d_closed, dtype=np.float64)
    L_o = np.asarray(lmk3d_open, dtype=np.float64)
    if adam_3d is None:
        raise ValueError("metrics_calculate 需要 ArUco 喉结点")
    if L_c.shape[0] < 68 or L_o.shape[0] < 68:
        raise ValueError("metrics_calculate 需要完整 68 点 landmark")

    def getp(V, key):
        i = IDX[key]
        return V[int(i)]

    def get_lmk(L, key):
        return L[int(LMK[key])]

    def get_brow(L):
        return 0.5 * (get_lmk(L, "brow_L") + get_lmk(L, "brow_R"))

    # 闭口点
    chin_c      = get_lmk(L_c, "chin")
    brow_c      = get_brow(L_c)
    upper_c     = get_lmk(L_c, "upper_lip")
    lower_c     = get_lmk(L_c, "lower_lip")
    adam_c      = np.asarray(adam_3d, dtype=np.float64)
    tragusL_c   = getp(V_c, "tragus_L")
    tragusR_c   = getp(V_c, "tragus_R")
    earlobeL_c  = getp(V_c, "earlobe_L")
    earlobeR_c  = getp(V_c, "earlobe_R")
    nose_c      = get_lmk(L_c, "nose_tip")

    # 张口点
    upper_o     = get_lmk(L_o, "upper_lip")
    lower_o     = get_lmk(L_o, "lower_lip")
    chin_o      = get_lmk(L_o, "chin")
    brow_o      = get_brow(L_o)

    SCALE = 100.0   # m → cm

    def to_cm(p):
        return p * 100.0

    chin_c     = to_cm(chin_c)
    brow_c     = to_cm(brow_c)
    upper_c    = to_cm(upper_c)
    lower_c    = to_cm(lower_c)
    adam_c     = to_cm(adam_c)
    tragusL_c  = to_cm(tragusL_c)
    tragusR_c  = to_cm(tragusR_c)
    earlobeL_c = to_cm(earlobeL_c)
    earlobeR_c = to_cm(earlobeR_c)
    nose_c     = to_cm(nose_c)
    upper_o    = to_cm(upper_o)
    lower_o    = to_cm(lower_o)
    chin_o     = to_cm(chin_o)
    brow_o     = to_cm(brow_o)

    # [1] 仰头角度
    # a 线：闭口位 下巴尖 → 眉心
    a_vec = brow_c - chin_c

    # b 线：闭口位 喉结 → 上唇中点（长度 b）
    b_vec = upper_c - adam_c
    b_len = np.linalg.norm(b_vec)

    head_tilt_angle = angle_between(a_vec, b_vec)

    # [2] 张口度
    # 张口度 = |上唇-下唇|_open - |上唇-下唇|_close
    mouth_opening = np.linalg.norm(upper_o - lower_o) - np.linalg.norm(upper_c - lower_c)

    # [3] 甲颏距离
    # 闭口位 下巴尖 → 喉结
    thyromental = np.linalg.norm(chin_c - adam_c)

    # [4] 下颌活动角度
    # 下巴标记线：下巴尖 → 下唇中点
    jawline_c = lower_c - chin_c
    jawline_o = lower_o - chin_o

    # 参考线：眉心 → 上唇中点
    ref_c = upper_c - brow_c
    ref_o = upper_o - brow_o

    jaw_angle_c = angle_between(jawline_c, ref_c)
    jaw_angle_o = angle_between(jawline_o, ref_o)

    jaw_activity_angle = np.abs(jaw_angle_o - jaw_angle_c)

    # [5] 下颌活动度
    jaw_activity_distance = (mouth_opening - 2.5) / 3.0 + jaw_activity_angle / 30.0

    # [6] 小下颌程度
    # 点：闭口位下巴尖
    # 线：喉结 → 上唇中点
    small_jaw_dist = point_to_line_distance(chin_c, adam_c, upper_c)

    if small_jaw_dist < 2.0:
        small_jaw_grade = "严重"
    elif small_jaw_dist < 4.0:
        small_jaw_grade = "轻度"
    else:
        small_jaw_grade = "正常"


    # [8] 舌厚度 t
    # t = b/2 + K
    K = 0
    tongue_thickness = b_len / 2.0 + K

    # [9] 声门视野计算
    # C 点：双侧耳屏中点
    C = 0.5 * (tragusL_c + tragusR_c)

    # E 点：双侧耳垂中点
    E = 0.5 * (earlobeL_c + earlobeR_c)

    # L 点：上唇中点（闭口位）
    L = upper_c

    # G 点：喉结
    G = adam_c

    # 旋转轴：左右耳屏连线方向
    axis_lr = tragusR_c - tragusL_c
    axis_lr = axis_lr / np.linalg.norm(axis_lr)

    # 旋转 12 度
    theta = np.radians(12.0)
    chin_rot = rodrigues_rotate(
        point=chin_c,
        center=C,
        axis_unit=axis_lr,
        theta_rad=theta
    )

    # 沿 C -> 鼻尖 方向前移（颞颌关节活动度）
    dir_forward = nose_c - C
    dir_forward = dir_forward / np.linalg.norm(dir_forward)

    M = chin_rot + dir_forward * jaw_activity_distance

    # 从 M 点指向耳垂中点 E，长度为 t/2
    dir_ME = E - M
    dir_ME = dir_ME / np.linalg.norm(dir_ME)

    T = M + dir_ME * (tongue_thickness / 2.0)

    # 视线：L -> T
    ray_origin = L
    ray_dir = T - L
    ray_dir_unit = ray_dir / np.linalg.norm(ray_dir)

    LG = G - L
    LG_unit = LG / np.linalg.norm(LG)

    LT = T - L
    LT_unit = LT / np.linalg.norm(LT)

    omega = np.arccos(np.dot(LG_unit, LT_unit)) # LGT平面夹角

    length = 2.5 - np.tan(omega) * np.linalg.norm(LG) # 视线与喉结距离


    # [7] 颈围（常量点 → PCA 平面 → 投影 → 椭圆拟合 → 长轴作直径）

    def fit_plane_pca(P):
        C = P.mean(axis=0)
        X = P - C
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        n = Vt[-1]
        n = n / np.linalg.norm(n)
        return C, n

    def project_to_plane(P, C, n):
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, n)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, tmp)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        Q = P - C
        return np.c_[Q @ e1, Q @ e2]

    def fit_ellipse_fitzgibbon(xy):
        """
        Fitzgibbon et al. (1999) Direct Least Squares Fitting of Ellipses.
        返回 (a, b): 半长轴, 半短轴 (a >= b).
        """
        x, y = xy[:, 0], xy[:, 1]
        D = np.c_[x*x, x*y, y*y, x, y, np.ones_like(x)]
        S = D.T @ D
        C1 = np.zeros((3, 3))
        C1[0, 2] = C1[2, 0] = 2.0
        C1[1, 1] = -1.0
        S11 = S[:3, :3]
        S12 = S[:3, 3:]
        S22 = S[3:, 3:]
        S22_inv = np.linalg.inv(S22)
        M = np.linalg.inv(C1) @ (S11 - S12 @ S22_inv @ S12.T)
        eigvals, eigvecs = np.linalg.eig(M)
        cond = 4 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2
        idx = np.argmax((cond > 0) & np.isfinite(eigvals))
        a1 = eigvecs[:, idx]
        a2 = -S22_inv @ S12.T @ a1
        coeffs = np.concatenate([a1, a2])  # A,B,C,D,E,F: Ax^2+Bxy+Cy^2+Dx+Ey+F=0
        A, B, C, D, E, F = coeffs
        det = B * B - 4 * A * C
        cx = (2 * C * D - B * E) / det
        cy = (2 * A * E - B * D) / det
        M_mat = np.array([[A, B / 2], [B / 2, C]])
        evals = np.linalg.eigvalsh(M_mat)
        num = -(A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F)
        axes = np.sqrt(num / evals)
        semi_a = max(axes)
        semi_b = min(axes)
        return semi_a, semi_b

    # 取闭口位颈围点
    P_ring = V_c[IDX["neck_ring"]] * SCALE   # cm

    C_plane, n_plane = fit_plane_pca(P_ring)
    xy_ring = project_to_plane(P_ring, C_plane, n_plane)

    semi_a, semi_b = fit_ellipse_fitzgibbon(xy_ring)
    # Ramanujan 二阶近似 (精度 ~1e-5, 远优于 π(a+b))
    h = ((semi_a - semi_b) / (semi_a + semi_b)) ** 2
    neck_circumference = math.pi * (semi_a + semi_b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

    return mouth_opening, thyromental, head_tilt_angle, jaw_activity_angle, jaw_activity_distance, neck_circumference, tongue_thickness, length
