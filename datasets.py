from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import open3d as o3d
import os
import sys
from plyfile import PlyData
import torch
import MinkowskiEngine as ME
import utils


class preprocessingDataset(Dataset):
    def __init__(self, root_data, mode, voxel, pointclouds="ply"):
        super(Dataset, self).__init__()
        self.root = root_data
        print("Root: ", self.root)
        self.directories = sorted(glob.glob('{}/*'.format(self.root)))
        print(self.directories)
        self.pcds = []
        self.mode = mode
        self.voxel_size = voxel
        print("Voxel size: ", self.voxel_size)
        if pointclouds == "ply":
            for i in self.directories:
                for j in glob.glob('{}/*.ply'.format(i)):
                        self.pcds.append(j)
        else:
            for i in self.directories:
                for j in glob.glob('{}/*.pcd'.format(i)):
                    self.pcds.append(j)
        

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):

        if self.mode == "train" or self.mode == "valid":
            pcd_raw = o3d.io.read_point_cloud(self.pcds[idx])
            self.coords = np.asarray(pcd_raw.points)
            self.coords = self.coords / self.voxel_size
            plydata = PlyData.read(self.pcds[idx])
            self.features=np.ones((self.coords.shape[0], 1))
            self.label = np.array((plydata.elements[0].data['labels']), dtype=np.int32)  # labels
            return self.coords, self.features, self.label


        else:
            pcd_raw = o3d.io.read_point_cloud(self.pcds[idx])
            points = np.asarray(pcd_raw.points)
            distances = np.linalg.norm(points, axis=1)
            umbral_dist = np.where((distances < 1) | (distances >= 45)) #remove points beyond 45m and closer than 1m
            new_points = np.delete(points, umbral_dist[0], axis=0)
            self.coords_rec = new_points
            self.coords = self.coords_rec/ self.voxel_size
            self.features=np.ones((self.coords.shape[0], 1))
            return self.coords, self.features
        
