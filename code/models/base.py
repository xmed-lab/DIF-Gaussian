import torch
import torch.nn as nn
import numpy as np

from models.utils import query_view_feats
from models.unet import UNet
from models.point_decoder import PointDecoder



class Recon_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init()

    def init(self):
        self.init_encoder()
        self.init_decoder(self.image_encoder.out_ch)
        self.registered_point_keys = ['points_proj']

    def init_encoder(self):
        cfg = self.cfg.image_encoder
        self.image_encoder = UNet(
            in_ch=1,
            out_ch=cfg.out_ch
        )
        
    def init_decoder(self, mid_ch):
        cfg = self.cfg.point_decoder
        self.point_decoder = PointDecoder(
            channels=[mid_ch] + cfg.mlp_chs,
            residual=True, 
            use_bn=True
        )

    def encode_projs(self, data):
        encoder_output = self.image_encoder(data['projs'])
        return {
            'feats_proj': encoder_output['feats']
        }
    
    def forward_points(self, feats_dict, data):
        # 1. multi-view pixel-aligned features + max-pooling
        p_feats = query_view_feats(
            view_feats=feats_dict['feats_proj'], 
            points_proj=data['points_proj'],
            fusion='max'
        )

        # 2. point-wise prediction
        p_pred = self.point_decoder(p_feats)
        return p_pred

    def forward(self, data, is_eval=False, eval_npoint=100000):
        feats_dict = self.encode_projs(data) # these features are shared for any sampled 3D points

        # point-wise forward
        if not is_eval:
            return {
                'points_pred': self.forward_points(feats_dict, data)
            }
        else:
            total_npoint = data['points_proj'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                
                tmp_data = {}
                for key in data.keys():
                    if key in self.registered_point_keys:
                        tmp_data[key] = data[key][..., left:right, :]
                    else: 
                        tmp_data[key] = data[key]
                
                points_pred = self.forward_points(feats_dict, tmp_data) # B, C, N
                pred_list.append(points_pred)

            return {
                'points_pred': torch.cat(pred_list, dim=2)
            }
