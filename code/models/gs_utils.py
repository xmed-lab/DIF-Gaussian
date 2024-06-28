import torch
import numpy as np
from pytorch3d.ops.knn import knn_points, knn_gather


def build_rotation(r):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py#L78
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    # r: [B, 4]
    # R: [B, 3, 3]
    
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + 
        r[:, 1] * r[:, 1] + 
        r[:, 2] * r[:, 2] + 
        r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # https://www.zhihu.com/tardis/zm/art/78987582?source_id=1003 (see Eqn. 21)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py#L101
    # s: [B, 3]
    # r: [B, 4]
    # L: [B, 3, 3]

    L = torch.zeros((s.size(0), 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance(s, r):
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py#L28
    # s: [B, N, 3]
    # r: [B, N, 4]
    # Cov: [B, N, 3, 3]

    b, n = s.shape[:2]
    s = s.reshape(b * n, 3)
    r = r.reshape(b * n, 4)
    
    L = build_scaling_rotation(s, r)
    Cov = L @ L.transpose(1, 2)

    Cov = Cov.reshape(b, n, 3, 3)
    return Cov


def query_gs(points, gs_points, gs_params):
    # points: [B, N, 3]
    # gs_points: [B, K, 3]
    # gs_params:
    #     - det: [B, K]
    #     - inv: [B, K, 3, 3]
    #     - offsets: [B, K, 3]
    #     - feats: [B, K, C]

    B, K = gs_points.shape[:2]
    N = points.shape[1]
    k = 3

    # 1. det and inv of Cov
    det = gs_params['det']
    inv = gs_params['inv']
    
    # 2. query neighbors
    # neb_idx: [B, N, k]
    # points_ext: [B, N, k, 3]
    # gs': [B, N, k, ...], (1) xyz; (2) det, inv; (3) feats.
    _, neb_idx, _ = knn_points(points, gs_points, K=k) # [B, N, k]
    points_ext = points.unsqueeze(2).repeat(1, 1, k, 1) # [B, N, k, 3]
    neb_gs_xyz = knn_gather(
        gs_points + gs_params['offsets'],
        neb_idx
    ) # [B, N, k, 3]
    neb_gs_feats = knn_gather(
        gs_params['feats'].transpose(1, 2), # [B, N, C]
        neb_idx
    ).permute(0, 3, 1, 2) # [B, N, k, C] => [B, C, N, k]
    neb_det = knn_gather(
        det.unsqueeze(-1),
        neb_idx
    ).squeeze(-1) # [B, N, k]
    neb_inv = knn_gather(
        inv.reshape(B, K, 9),
        neb_idx
    ).reshape(B, N, k, 3, 3) # [B, N, k, 3, 3]

    # 3. calculate weights
    # let * be [B, N, k]
    # weights: [*]
    # w = (2*pi)^(-k/2) * det^(-1/2) * exp(-1/2 * (p-mu)' * inv * (p-mu))
    neb_gs_xyz = (points_ext - neb_gs_xyz).unsqueeze(-1) # [*, 3, 1]
    weights = np.power(2 * np.pi, -3/2) * \
        torch.pow(neb_det, -1/2) * \
        torch.exp(
            -1/2 * neb_gs_xyz.transpose(3, 4) @ neb_inv @ neb_gs_xyz # [*, 1, 3] @ [*, 3, 3] @ [*, 3, 1] => [*, 1, 1]
        ).squeeze(4).squeeze(3) # [*]

    # 4. sum weighted features
    # weights * gs_feats' =(sum)> [B, C, N]
    weights = weights.unsqueeze(1) # [B, 1, N, k]
    sum_feats = (weights * neb_gs_feats).sum(dim=-1) # [B, C, N]

    return sum_feats
