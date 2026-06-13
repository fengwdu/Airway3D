import torch

def umeyama_similarity(X, Y, eps=1e-9):      # finished
    """
    X: [N, 3]
    Y: [N, 3]
    """
    N = X.shape[0]

    X_mean = torch.mean(X, dim=0, keepdim=True)
    Y_mean = torch.mean(Y, dim=0, keepdim=True)

    X_c = X - X_mean
    Y_c = Y - Y_mean

    sigma = Y_c.T @ X_c / N

    U, S, Vt = torch.linalg.svd(sigma)

    D = torch.ones(3, device=X.device, dtype=X.dtype)
    if torch.det(U) * torch.det(Vt) < 0:
        D[-1] = -1

    R = U @ torch.diag(D) @ Vt

    var_x = (X_c ** 2).sum() / N
    scale = (S * D).sum() / (var_x + eps)

    t = Y_mean.squeeze(0) - scale * (R @ X_mean.squeeze(0))

    return scale, R, t

def AA_to_matrix(aa):
    """
    把轴角转换成旋转矩阵
    axis: tensor (3,) -> (3,3)
    """
    theta = torch.linalg.norm(aa)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype)

    axis = aa / (theta + 1e-8)
    x, y, z = axis[0], axis[1], axis[2]
    zero = torch.zeros_like(x)
    
    K = torch.stack([
        torch.stack([ zero, -z,    y ]),
        torch.stack([ z,    zero, -x ]),
        torch.stack([-y,     x,   zero]),
    ], dim=0)

    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R

def AA_to_matrix_batch(aa):
    """
    aa: (B,3) axis-angle rotation vector
    return: (B,3,3)
    """
    B = aa.shape[0]
    theta = torch.linalg.norm(aa, dim=1, keepdim=True)  # (B,1)
    # avoid div0; for tiny theta, use identity approx
    axis = aa / (theta + 1e-8)  # (B,3)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)

    K = torch.stack([
        torch.stack([ zero, -z,    y ], dim=1),
        torch.stack([ z,    zero, -x ], dim=1),
        torch.stack([-y,     x,   zero], dim=1),
    ], dim=1)  # (B,3,3)

    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0).expand(B, -1, -1)
    sin_t = torch.sin(theta).view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)

    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)

    return R

def matrix_to_AA(R):
    """
    R: [3, 3] matrix
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = cos_theta.clamp(-1, 1)
    theta = torch.acos(cos_theta)

    if theta < 1e-8:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    
    if torch.abs(theta - torch.pi) < 1e-4:
        # 180° 情况，用对角线恢复轴
        axis = torch.sqrt(torch.clamp((torch.diag(R) + 1) / 2, min=0))
        axis = axis / (axis.norm() + 1e-8)
        return axis * theta

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]    

    axis = torch.stack([rx, ry, rz]) / (2 * torch.sin(theta))

    return axis * theta

@torch.no_grad()
def closest_point_barycentric(P, A, B, C, eps=1e-12):
    """
    P: (K,3)
    A,B,C: (K,3)
    return:
      w: (K,3) barycentric weights for closest point on triangle
      Q: (K,3) closest point
      dist2: (K,) squared distance
    Reference: Christer Ericson, Real-Time Collision Detection (triangle closest point)
    """
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = (AB * AP).sum(dim=1)
    d2 = (AC * AP).sum(dim=1)

    # Region A
    is_A = (d1 <= 0) & (d2 <= 0)

    BP = P - B
    d3 = (AB * BP).sum(dim=1)
    d4 = (AC * BP).sum(dim=1)
    # Region B
    is_B = (d3 >= 0) & (d4 <= d3)

    # Edge AB
    vc = d1 * d4 - d3 * d2
    is_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_AB = d1 / (d1 - d3 + eps)

    CP = P - C
    d5 = (AB * CP).sum(dim=1)
    d6 = (AC * CP).sum(dim=1)
    # Region C
    is_C = (d6 >= 0) & (d5 <= d6)

    # Edge AC
    vb = d5 * d2 - d1 * d6
    is_AC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_AC = d2 / (d2 - d6 + eps)

    # Edge BC
    va = d3 * d6 - d5 * d4
    is_BC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_BC = (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps)

    # Inside face
    is_face = ~(is_A | is_B | is_C | is_AB | is_AC | is_BC)

    w0 = torch.zeros(P.shape[0], device=P.device, dtype=P.dtype)
    w1 = torch.zeros_like(w0)
    w2 = torch.zeros_like(w0)

    # A
    w0[is_A] = 1.0
    # B
    w1[is_B] = 1.0
    # C
    w2[is_C] = 1.0

    # AB: Q = A + v*AB => bary = (1-v, v, 0)
    w0[is_AB] = 1.0 - v_AB[is_AB]
    w1[is_AB] = v_AB[is_AB]

    # AC: Q = A + w*AC => bary = (1-w, 0, w)
    w0[is_AC] = 1.0 - w_AC[is_AC]
    w2[is_AC] = w_AC[is_AC]

    # BC: Q = B + w*(C-B) => bary = (0, 1-w, w)
    w1[is_BC] = 1.0 - w_BC[is_BC]
    w2[is_BC] = w_BC[is_BC]

    # Face: compute barycentric for projection onto triangle plane
    if is_face.any():
        denom = (va + vb + vc + eps)
        v = vb / denom
        w = vc / denom
        u = 1.0 - v - w
        w0[is_face] = u[is_face]
        w1[is_face] = v[is_face]
        w2[is_face] = w[is_face]

    W = torch.stack([w0, w1, w2], dim=1)  # (K,3)
    Q = W[:, 0:1] * A + W[:, 1:2] * B + W[:, 2:3] * C
    dist2 = ((P - Q) ** 2).sum(dim=1)
    return W, Q, dist2

@torch.no_grad()
def build_face_candidates(points, verts, verts2face, knn_k=5, batch_size=10000):
    """
    points: (N,3)
    verts:  (V,3)
    verts2face: (V,10)  each vertex -> up to 10 incident faces, -1 padded
    return:
      face_cand: (N, C) long, C=knn_k*10, -1 padded
    """
    device = points.device
    N = points.shape[0]
    chunks = []
    for i in range(0, N, batch_size):
        P = points[i:i+batch_size]                  # (B,3)
        d = torch.cdist(P, verts)                   # (B,V)
        _, knn_idx = torch.topk(d, k=knn_k, dim=1, largest=False)  # (B,knn_k)
        face_idx = verts2face[knn_idx]              # (B,knn_k,10)
        face_idx = face_idx.reshape(P.shape[0], -1) # (B,C)
        chunks.append(face_idx)
    face_cand = torch.cat(chunks, dim=0)            # (N,C)
    return face_cand

@torch.no_grad()
def build_point_plane_winner(points, verts, faces, face_cand, eps=1e-12):
    """
    points: (N,3)
    verts:  (V,3)
    faces:  (F,3) long
    face_cand: (N,C) long, -1 padded
    return (cache):
      face_id: (N,) long   winner face index, -1 if none
      bary_w:  (N,3) float barycentric weights of closest point on that face
      normal:  (N,3) float unit normal of that face (frozen)
      valid:   (N,) bool
    """
    device = points.device
    dtype = points.dtype
    N, C = face_cand.shape

    valid_c = face_cand >= 0                         # (N,C)
    # Replace invalid with 0 to allow gather; will be masked out by valid_c later
    safe_face = torch.where(valid_c, face_cand, torch.zeros_like(face_cand))

    tri = verts[faces[safe_face]]                    # (N,C,3,3)
    A = tri[:, :, 0, :]                              # (N,C,3)
    B = tri[:, :, 1, :]
    Cc = tri[:, :, 2, :]
    P = points[:, None, :].expand(N, C, 3)           # (N,C,3)

    # face normal (unit)
    n = torch.cross(B - A, Cc - A, dim=2)           # (N,C,3)
    n_norm = torch.linalg.norm(n, dim=2, keepdim=True).clamp_min(eps)
    n = n / n_norm                                   # (N,C,3)

    # closest point + barycentric (vectorized by flatten then reshape)
    Pf = P.reshape(-1, 3)
    Af = A.reshape(-1, 3)
    Bf = B.reshape(-1, 3)
    Cf = Cc.reshape(-1, 3)

    Wf, Qf, _ = closest_point_barycentric(Pf, Af, Bf, Cf, eps=eps)  # (N*C,3),(N*C,3)
    W = Wf.view(N, C, 3)
    Q = Qf.view(N, C, 3)

    r = ((P - Q) * n).sum(dim=2)                     # (N,C) point-to-plane residual
    val = r * r                                      # (N,C)

    # Mask invalid candidates as +inf so argmin ignores them
    val = torch.where(valid_c, val, torch.full_like(val, float("inf")))

    best_j = torch.argmin(val, dim=1)                # (N,)
    best_val = val[torch.arange(N, device=device), best_j]
    valid = torch.isfinite(best_val)

    face_id = face_cand[torch.arange(N, device=device), best_j]
    bary_w  = W[torch.arange(N, device=device), best_j, :]          # (N,3)
    normal  = n[torch.arange(N, device=device), best_j, :]          # (N,3)

    # invalidate entries
    face_id = torch.where(valid, face_id, torch.full_like(face_id, -1))
    return face_id, bary_w, normal, valid


def point_to_plane_loss_from_cache(points, verts, faces, cache, robust_eps=1e-6):
    """
    Differentiable inner loss with fixed winners.
    points: (N,3)
    verts:  (V,3) requires_grad
    cache: (face_id, bary_w, normal, valid)
    return: scalar loss
    """
    face_id, bary_w, normal, valid = cache
    if valid.sum().item() == 0:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)

    pid = torch.nonzero(valid, as_tuple=False).squeeze(1)       # (M,)
    fid = face_id[pid]                                          # (M,)
    W   = bary_w[pid]                                           # (M,3)
    n   = normal[pid]                                           # (M,3)
    P   = points[pid]                                           # (M,3)

    tri = verts[faces[fid]]                                     # (M,3,3)
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    Q = W[:, 0:1] * A + W[:, 1:2] * B + W[:, 2:3] * C           # (M,3)

    r = (n * (P - Q)).sum(dim=1)                                # (M,)

    # Huber-like (simple, stable): sqrt(r^2 + eps)
    loss = torch.sqrt(r * r + robust_eps).mean()
    return loss
