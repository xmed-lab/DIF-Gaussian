import os
import json
import yaml
import scipy
import pickle
import numpy as np
from torch.utils.data import Dataset

from utils import sitk_load
from datasets.geometry import Geometry



class CBCT_dataset(Dataset):
    def __init__(
            self,
            dst_name,
            cfg,
            split='train',
            num_views=10,
            npoint=5000,
            out_res_scale=1.0,
            random_views=False,
            view_offset=0,
        ):
        super().__init__()
        if not (0. < out_res_scale and out_res_scale <= 1.):
            raise ValueError
        
        self.cfg = cfg
        self.dst_name = dst_name
        self.data_root = os.path.join(self.cfg.root_dir, dst_name)
        
        # load dataset info
        with open(os.path.join(self.data_root, 'meta_info.json'), 'r') as f:
            self.info = json.load(f)
            splits = split.split('+')
            name_list = []
            for s in splits:
                name_list += self.info[s]
            name_list = sorted(name_list)
            print('CBCT_dataset, name: {}, split: {}, len: {}.'.format(dst_name, split, len(name_list)))

        # load dataset config
        with open(os.path.join(self.data_root, self.info['dataset_config']), 'r') as f:
            dst_cfg = yaml.safe_load(f)
            out_res = np.array(dst_cfg['dataset']['resolution'])
            out_res = np.round(out_res * out_res_scale).astype(int) # to align the output shape with 'scipy.ndimage.zoom'
            self.geo = Geometry(dst_cfg['projector'])

        # prepare points
        if split == 'train':
            # load blocks' coordinates [train only]
            self.blocks = np.load(os.path.join(self.data_root, self.info['blocks_coords']))
        else:
            # prepare sampling points
            points = np.mgrid[:out_res[0], :out_res[1], :out_res[2]]
            points = points.astype(np.float32)
            points = points.reshape(3, -1)
            points = points.transpose(1, 0) # N, 3
            self.points = points / (out_res - 1)
        
        # other parameters
        self.out_res_scale = out_res_scale
        self.name_list = name_list
        self.npoint = npoint
        self.is_train = (split == 'train')
        self.num_views = num_views
        self.random_views = random_views
        self.view_offset = view_offset

        # for acceleration when testing
        self.points_proj = None

    def __len__(self):
        return len(self.name_list)
    
    def sample_projections(self, name, n_view=None):
        # -- load projections
        with open(os.path.join(self.data_root, self.info['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']         # uint8: [K, W, H]
            projs_max = data['projs_max'] # float
            angles = data['angles']       # float: [K,]

        if n_view is None:
            n_view = self.num_views

        # -- sample projections
        views = np.linspace(0, len(projs), n_view, endpoint=False).astype(int) # endpoint=False as the random_views is True during training, i.e., enabling view offsets.
        offset = np.random.randint(len(projs) - views[-1]) if self.random_views else self.view_offset
        views += offset

        projs = projs[views].astype(np.float32) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]

        # -- de-normalization
        projs = projs * projs_max / 0.2

        return projs, angles
    
    def load_ct(self, name):
        image, spacing, origin = sitk_load(
            os.path.join(self.data_root, self.info['image'].format(name)),
            uint8=True
        ) # float32
        if self.out_res_scale < 1.:
            image = scipy.ndimage.zoom(image, self.out_res_scale, order=3, prefilter=False)
            spacing /= self.out_res_scale
        return image, spacing, origin
    
    def load_block(self, name, b_idx):
        path = os.path.join(self.data_root, self.info['blocks_vals'].format(name, b_idx))
        block = np.load(path) # uint8
        return block

    def sample_points(self, points, values):
        choice = np.random.choice(len(points), size=self.npoint, replace=False)
        points = points[choice]
        values = values[choice]
        return points, values

    def project_points(self, points, angles):
        points_proj = []
        for a in angles:
            p = self.geo.project(points, a)
            points_proj.append(p)
        points_proj = np.stack(points_proj, axis=0) # [M, N, 2]
        return points_proj

    def __getitem__(self, index):
        name = self.name_list[index]

        # -- load projections
        projs, angles = self.sample_projections(name)

        # -- load sampling points
        if not self.is_train:
            points = self.points
            points_gt, spacing, origin = self.load_ct(name)
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            block_coords = self.blocks[b_idx] # [N, 3]
            points, points_gt = self.sample_points(block_coords, block_values)
            points_gt = points_gt.astype(np.float32) / 255.
            points_gt = points_gt[None, :]

        # -- project points
        if self.is_train or self.points_proj is None:
            points_proj = self.project_points(points, angles) # given the same geo cfg
            self.points_proj = points_proj
        else:
            points_proj = self.points_proj

        # -- collect data
        ret_dict = {
            # M: the number of views
            # N: the number of sampled points
            'dst_name': self.dst_name,
            'name': name,
            'angles': angles,           # [M,]
            'projs': projs,             # [M, 1, W, H], projections
            'points': points,           # [N, 3], center xyz of volumes ~[0, 1]
            'points_gt': points_gt,     # [1, N] (or [W', H', D'] only when is_train is False)
            'points_proj': points_proj, # [M, N, 2]
        }
        if not self.is_train:
            ret_dict['spacing'] = spacing # [3,]
            ret_dict['origin'] = origin   # [3,]
        return ret_dict
