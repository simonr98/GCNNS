import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
from definitions import ROOT_DIR, INDEX_TRACK_POINT_VOXEL, INDEX_TRAJECTORY_VOXEL, CAMERA_COORDINATES_TORUS
from typing import Optional


# ---------------------------------------------------------------------------------------------------------------------
# Voxel Data Processing

@dataclass
class VoxelData:
    nodes: np.ndarray
    pcd: np.ndarray
    y: np.ndarray


def get_train_data_voxel():
    with open(f"{ROOT_DIR}/data/voxel/coarse_60steps_voxel_color_train.pkl", "rb") as input_file:
        train_data = pickle.load(input_file)
    return train_data


def get_test_data_voxel():
    with open(f"{ROOT_DIR}/data/voxel/coarse_60steps_voxel_color_test.pkl", "rb") as input_file:
        test_data = pickle.load(input_file)
    return test_data


def get_one_voxel_trajectory(train_data: Optional[list], index_t: int, index_poi: int, num_points_point_cloud=60):
    data_t = train_data[index_t]
    nodes, pcd, y = [], [], []

    for p_pcd, p_node in zip(data_t['pcd_points'], data_t['nodes_grid']):
        random = np.random.choice(p_pcd.shape[0], num_points_point_cloud, replace=False)
        pcd.append(p_pcd[random])

        nodes.append(p_node)
        y.append(p_node[index_poi])

    nodes, pcd, y = np.array(nodes), np.array(pcd), np.array(y)

    return VoxelData(nodes, pcd, y)


def get_voxel_data(data: Optional[list], target_ind: int, num_points_point_cloud=60):
    nodes = [get_one_voxel_trajectory(data, index, target_ind, num_points_point_cloud).nodes for index in
             range(len(data))]
    pcd = [get_one_voxel_trajectory(data, index, target_ind, num_points_point_cloud).pcd for index in range(len(data))]
    y = [get_one_voxel_trajectory(data, index, target_ind, num_points_point_cloud).y for index in range(len(data))]

    return VoxelData(np.array(nodes), np.array(pcd), np.array(y))


# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Torus data processing

@dataclass
class TorusData:
    com: np.ndarray
    pcd: np.ndarray
    pos: np.ndarray


def get_torus_data(test=False):
    mode = 'test' if test else 'train'

    with open(f"{ROOT_DIR}/data/torus/{mode}_data.pkl", "rb") as input_file:
        data = pickle.load(input_file)

    com, pcd, pos = np.array(data['com']), np.array(data['pcd:']), np.array(data['pos'])

    # Only take t trajectories for training and 2 for testing (as each pc has ~900 points)
    i = 6 if mode == 'train' else 2

    com, pcd, pos = com[:i, :, :], pcd[:i, :, :, :], pos[:i, :, :]

    return TorusData(com, pcd, pos)


# ---------------------------------------------------------------------------------------------------------------------

def join_trajectories(data):
    return np.vstack([data[i] for i in range(len(data))])


##########################################################################################################
# TESTS
##########################################################################################################

if __name__ == '__main__':
    test_one_trajectory = False
    test_data = False
    test_all_trajectories = False
    test_join_trajectories = False
    test_torus_data = True

    train_data = get_train_data_voxel()
    # --------------------------------------------------------------------------------------------------
    # TEST 1 - TEST THE DATA - Points can move but cannot jump
    if test_data:
        nodes_grid = train_data[1]['nodes_grid']

        nodes_grid = np.array([p[0] for p in nodes_grid])

        plt.scatter(nodes_grid[:, 0], nodes_grid[:, 1])
        plt.show()

    # --------------------------------------------------------------------------------------------------
    # TEST 2 - TEST get_one_trajectory - nodes should be consistent
    voxel_data = get_one_voxel_trajectory(train_data, INDEX_TRAJECTORY_VOXEL, INDEX_TRACK_POINT_VOXEL)
    nodes = voxel_data.nodes

    if test_one_trajectory:
        nodes = nodes[:, 0, :]
        plt.scatter(nodes[:, 0], nodes[:, 1])
        plt.show()

    # --------------------------------------------------------------------------------------------------
    # TEST 3 - TEST get_all_trajectories - nodes should be consistent
    voxel_data = get_voxel_data(train_data, INDEX_TRACK_POINT_VOXEL)
    nodes = voxel_data.nodes

    if test_all_trajectories:
        nodes = nodes[0, :, 0, :]
        plt.scatter(nodes[:, 0], nodes[:, 1])
        plt.show()

    # --------------------------------------------------------------------------------------------------
    # TEST 4 - TEST join_trajectories
    voxel_data = get_voxel_data(train_data, 50)
    nodes, pcd, y = voxel_data.nodes, voxel_data.pcd, voxel_data.y

    if test_join_trajectories:
        print(join_trajectories(pcd).shape)
        print(join_trajectories(nodes).shape)
        print(join_trajectories(y).shape)

    # --------------------------------------------------------------------------------------------------
    # TEST 5 - TEST get_torus_data
    torus_data = get_torus_data()
    com, pcd, pos = torus_data.com, torus_data.pcd, torus_data.pos

    if test_torus_data:
        print(com.shape)
        print(pcd.shape)
        print(pos.shape)

        # --------------------------------------------------------------------------------------------------------
        # Plot Point Cloud with noise
        fig = plt.figure(1, figsize=(100, 100))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Point cloud with noise', {'fontsize': 30})
        ax.scatter(pcd[0, 0, :, 0], pcd[0, 0, :, 1], pcd[0, 0, :, 2], c='red')

        x, y, z = CAMERA_COORDINATES_TORUS[0], CAMERA_COORDINATES_TORUS[1], CAMERA_COORDINATES_TORUS[2]
        plt.plot([x], [y], [z], marker='o', markersize=10, color="black")
        # ----------------------------------------------------------------------------------------------------------

        plt.subplots_adjust()
        plt.show()

##########################################################################################################
