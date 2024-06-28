import torch
import torch.nn as nn
import numpy as np

from models.base import Recon_base
from models.utils import query_view_feats, MLP_1d
from models.gs_utils import query_gs, build_covariance



class DIF_Gaussian(Recon_base):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self):
        self.init_encoder()
        
        # gaussians-related modules
        mid_ch = self.image_encoder.out_ch
        ds_ch = self.image_encoder.ds_ch
        self.gs_feats_mlp = MLP_1d([ds_ch, ds_ch // 4, mid_ch], use_bn=True, last_bn=True, last_act=False)
        self.gs_params_mlp = MLP_1d([ds_ch, ds_ch // 4, 3 + 4 + 3], use_bn=True, last_bn=False, last_act=False) # 3d: offsets, 4d: rotation, 3d: scaling
        self.gs_act = nn.LeakyReLU(inplace=True)

        self.init_decoder(mid_ch * 2)
        self.registered_point_keys = ['points', 'points_proj']

    def encode_projs(self, data):
        encoder_output = self.image_encoder(data['projs'])

        p_feats = query_view_feats(
            view_feats=encoder_output['feats_ds'],
            points_proj=data['points_gs_proj'],
            fusion='max'
        )

        # features
        gs_feats = self.gs_feats_mlp(p_feats)
        
        # hyper-parameters
        gs_cfg = self.cfg.gs

        t = 1. / gs_cfg.t
        gs_res = gs_cfg.res
        p_dist_scaling = gs_cfg.p_dist_scaling
        o_scaling = gs_cfg.o_scaling
        s_scaling = gs_cfg.s_scaling

        p_dist = 1 / gs_res * p_dist_scaling
        
        # other gs parameters
        gs_params = self.gs_params_mlp(p_feats)
        # gs_params[:, :6, :] = gs_params[:, :6, :] / t
        gs_params = gs_params / t

        # offsets
        max_dist = p_dist * 0.5 * o_scaling # 0.5: half (two neighboring points share the distance)
        offsets = gs_params[:, :3, :].transpose(1, 2) # [B, K, 3]
        offsets = (torch.sigmoid(offsets) - 0.5) * 2 * max_dist

        # s
        s_mean = np.sqrt(p_dist)
        s = gs_params[:, 3:6, :].transpose(1, 2) # [B, K, 3]
        s = torch.sigmoid(s) * 2 * s_scaling * s_mean + (1 - s_scaling) * s_mean # (0, 1) => ((1-k)d, (1+k)d)

        # r
        r = gs_params[:, 6:, :].transpose(1, 2) # [B, K, 4]

        # convariance matrix, det/inv
        Cov = build_covariance(s, r) # [B, K, 3, 3]
        det = torch.linalg.det(Cov)  # [B, K]
        inv = torch.linalg.inv(Cov)  # [B, K, 3, 3]

        return {
            'feats_proj': encoder_output['feats'],
            'gs_params': {
                'feats': gs_feats,  # [B, C, K]
                'offsets': offsets, # [B, K, 3]
                'det': det,         # [B, K]
                'inv': inv,         # [B, K, 3, 3]
            }
        }

    def forward_points(self, feats_dict, data):
        # 1. multi-view pixel-aligned features + max-pooling
        p_feats = query_view_feats(
            view_feats=feats_dict['feats_proj'], 
            points_proj=data['points_proj'],
            fusion='max'
        )

        # 2. gaussian-based interp
        interp_feats = query_gs(
            data['points'], 
            data['points_gs'], 
            feats_dict['gs_params']
        )
        interp_feats = self.gs_act(interp_feats)
        p_feats = torch.cat([p_feats, interp_feats], dim=1)

        # 3. point-wise prediction
        p_pred = self.point_decoder(p_feats)
        return p_pred
