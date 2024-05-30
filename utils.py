import glob
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


def normalize_features(features, minimum, maximum):
    norm_arr = np.empty_like(features)
    # for dim in range(features.shape[1]): # In case of multiple features
    diff_arr = maximum - minimum
    for n, l in enumerate(features[:]):
        norm_arr[n] = ((l - minimum) / diff_arr)
    return norm_arr.astype(np.float32)

def compute_normals(pcd):
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamKNN(6))
    pcd.orient_normals_to_align_with_direction()
    normals = np.asarray(pcd.normals)
    ey = o3d.geometry.PointCloud()
    ey.points = o3d.utility.Vector3dVector(pcd.points)
    ey.normals = o3d.utility.Vector3dVector(normals)
    return ey

def assign_feats(sp, x):
    return ME.SparseTensor(
        features=x.float(),
        coordinate_map_key=sp.coordinate_map_key,
        coordinate_manager=sp.coordinate_manager,
    )


class SparseTensorLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, sp):
        x = self.norm(sp.F)
        return assign_feats(sp, x.float())

class SparseTensorLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, sp):
        x = self.linear(sp.F)
        return assign_feats(sp, x.float())