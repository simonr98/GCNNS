import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from definitions import ROOT_DIR
from util.VoxelDataset import VoxelDataset
from util.preprocessing import get_all_voxel_trajectories, join_trajectories, get_train_data, get_test_data


def get_model_net_data(num_points_pc: int, batch_size: int):

    path = osp.join(ROOT_DIR, 'data/ModelNet10')

    # Transform Mesh to Point Cloud
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(num_points_pc)

    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    return train_loader, test_loader


def get_voxel_data(num_points_pc: int, track_point_index: int):

    train_data, test_data = get_train_data(), get_test_data()

    _, pcd_train, y_train = get_all_voxel_trajectories(train_data, track_point_index, num_points_pc)
    pcd_train, y_train = join_trajectories(pcd_train), join_trajectories(y_train)

    _, pcd_test, y_test = get_all_voxel_trajectories(test_data, track_point_index, num_points_pc)
    pcd_test, y_test = join_trajectories(pcd_test), join_trajectories(y_test)

    train_dataset = VoxelDataset(data=pcd_train, targets=y_train)
    test_dataset = VoxelDataset(data=pcd_test, targets=y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

    return train_loader, test_loader


