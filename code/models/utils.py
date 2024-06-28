import torch
from torch import nn
import torch.nn.functional as F



def index_3d(feat, uv):
    '''
    :param feat: [B, C, H, W, D] image features
    :param uv: [B, N, 3] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]; 5-d case
    feat = feat.transpose(2, 4)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1, 1]
    return samples[:, :, :, 0, 0]  # [B, C, N]


def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]


def query_view_feats(view_feats, points_proj, fusion='max'):
    # view_feats: [B, M, C, H, W]
    # points_proj: [B, M, N, 2]
    # output: [B, C, N, M]
    n_view = view_feats.shape[1]
    p_feats_list = []
    for i in range(n_view):
        feat = view_feats[:, i, ...] # B, C, W, H
        p = points_proj[:, i, ...] # B, N, 2
        p_feats = index_2d(feat, p) # B, C, N
        p_feats_list.append(p_feats)
    p_feats = torch.stack(p_feats_list, dim=-1) # B, C, N, M
    if fusion == 'max':
        p_feats = F.max_pool2d(p_feats, (1, p_feats.shape[-1]))
        p_feats = p_feats.squeeze(-1) # [B, C, K]
    elif fusion is not None:
        raise NotImplementedError
    return p_feats


class MLP_1d(nn.Module):
    def __init__(self, mlp_list, use_bn=False, last_bn=True, last_act=True):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv1d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn and (last_bn or i < len(mlp_list) - 2):
                layers += [nn.BatchNorm1d(mlp_list[i + 1])]
            if last_act or i < len(mlp_list) - 2:
                layers += [nn.LeakyReLU(inplace=True)]
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
