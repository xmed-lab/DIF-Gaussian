import numpy as np
from copy import deepcopy

from datasets.base import CBCT_dataset



class CBCT_dataset_gs(CBCT_dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gs_res = self.cfg.gs_res
        points_gs = np.mgrid[:gs_res, :gs_res, :gs_res] / gs_res
        self.points_gs = points_gs.reshape(3, -1).transpose(1, 0) # ~[0, 1]

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        # projections of GS points (initial center xyz)
        points_gs = deepcopy(self.points_gs)
        points_gs_proj = self.project_points(points_gs, data_dict['angles'])

        data_dict.update({
            'points_gs': points_gs,          # [K, 3]
            'points_gs_proj': points_gs_proj # [M, K, 2]
        })
        return data_dict
