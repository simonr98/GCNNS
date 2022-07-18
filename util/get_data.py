import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from definitions import ROOT_DIR
from util.CustomDataset import CustomDataset
from util.preprocessing import get_voxel_data, join_trajectories, get_train_data_voxel, get_test_data_voxel, \
    get_torus_data


def get_model_net_data(num_points_pc: int, batch_size: int):

    path = osp.join(ROOT_DIR, 'data/ModelNet10')

    # Transform Mesh to Point Cloud
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(num_points_pc)

    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    return train_loader, test_loader


def get_voxel_data_loaders(num_points_pc: int, track_point_index: int):
    train_data, test_data = get_train_data_voxel(), get_test_data_voxel()

    voxel_data_train = get_voxel_data(train_data, track_point_index, num_points_pc)
    pcd_train, y_train = voxel_data_train.pcd, voxel_data_train.y
    pcd_train, y_train = join_trajectories(pcd_train), join_trajectories(y_train)

    voxel_data_test = get_voxel_data(test_data, track_point_index, num_points_pc)
    pcd_test, y_test = voxel_data_test.pcd, voxel_data_test.y
    pcd_test, y_test = join_trajectories(pcd_test), join_trajectories(y_test)

    train_dataset = CustomDataset(data=pcd_train, targets=y_train)
    test_dataset = CustomDataset(data=pcd_test, targets=y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

    return train_loader, test_loader


def get_torus_data_loaders(com: bool = True, key ='pcd'):
    train_data, test_data = get_torus_data(test=False, key=key), get_torus_data(test=True, key=key)

    if com:
        y_train, y_test = train_data.com, test_data.com
    else:
        y_train, y_test = train_data.pos, test_data.pos

    input_train, input_test = train_data.input, test_data.input

    input_train, y_train = join_trajectories(input_train), join_trajectories(y_train)
    input_test, y_test = join_trajectories(input_test), join_trajectories(y_test)

    train_dataset = CustomDataset(data=input_train, targets=y_train)
    test_dataset = CustomDataset(data=input_test, targets=y_test)

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=6)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_model_net_data(num_points_pc=480, batch_size=32)

    for data in train_loader:
        print(data.pos.shape)
