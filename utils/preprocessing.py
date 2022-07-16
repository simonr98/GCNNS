import numpy as np
import pickle
import matplotlib.pyplot as plt

with open(r"data/voxel/coarse_60steps_voxel_color_train.pkl", "rb") as input_file:
    train_data = pickle.load(input_file)

with open(r"data/voxel/coarse_60steps_voxel_color_test.pkl", "rb") as input_file:
    test_data = pickle.load(input_file)


def get_train_data():
    return train_data


def get_test_data():
    return test_data


def get_one_trajectory(train_data: [], index_t: int, index_poi: int, num_points_point_cloud=60):

    data_t = train_data[index_t]
    nodes, pcd, y = [], [], []

    for p_pcd, p_node in zip(data_t['pcd_points'], data_t['nodes_grid']):
        random = np.random.choice(p_pcd.shape[0], num_points_point_cloud, replace=False)
        pcd.append(p_pcd[random])

        nodes.append(p_node)
        y.append(p_node[index_poi])

    nodes, pcd, y = np.array(nodes), np.array(pcd), np.array(y)

    return nodes, pcd, y


def get_all_trajectories(data: [], target_ind: int, num_points_point_cloud=60):
    nodes = [get_one_trajectory(data, index, target_ind, num_points_point_cloud)[0] for index in range(len(data))]
    pcd = [get_one_trajectory(data, index, target_ind, num_points_point_cloud)[1] for index in range(len(data))]
    y = [get_one_trajectory(data, index, target_ind, num_points_point_cloud)[2] for index in range(len(data))]

    return np.array(nodes), np.array(pcd), np.array(y)


def join_trajectories(data):
    return np.vstack([data[i] for i in range(len(data))])


##########################################################################################################
# TEST
##########################################################################################################

if __name__ == '__main__':

    test_one_trajectory = False
    test_data = False
    test_all_trajectories = False
    test_join_trajectories = False

    # --------------------------------------------------------------------------------------------------
    # TEST 1 - TEST THE DATA - Points can move but cannot jump
    if test_data:
        nodes_grid = train_data[1]['nodes_grid']

        nodes_grid = np.array([p[0] for p in nodes_grid])

        plt.scatter(nodes_grid[:, 0], nodes_grid[:, 1])
        plt.show()

    # --------------------------------------------------------------------------------------------------
    # TEST 2 - TEST get_one_trajectory - nodes should be consistent
    nodes, pcd, y = get_one_trajectory(train_data, 0, 50)

    if test_one_trajectory:
        nodes = nodes[:, 0, :]
        plt.scatter(nodes[:, 0], nodes[:, 1])
        plt.show()


    # --------------------------------------------------------------------------------------------------
    # TEST 3 - TEST get_all_trajectories - nodes should be consistent
    nodes, pcd, y = get_all_trajectories(train_data, 50)

    if test_all_trajectories:
        nodes = nodes[0, :, 0, :]
        plt.scatter(nodes[:, 0], nodes[:, 1])
        plt.show()

    if test_join_trajectories:
        print(join_trajectories(pcd).shape)
        print(join_trajectories(nodes).shape)
        print(join_trajectories(y).shape)
