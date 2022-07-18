import torch as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from definitions import INDEX_TRACK_POINT_VOXEL, NUM_POINTS_POINT_CLOUD
from util.preprocessing import *


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = T.from_numpy(data).float()
        self.targets = T.FloatTensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


##########################################################################################################
# TEST
##########################################################################################################
if __name__ == '__main__':
    voxel_train_data = get_voxel_data(get_train_data_voxel(), INDEX_TRACK_POINT_VOXEL, NUM_POINTS_POINT_CLOUD)
    nodes, pcd, target = voxel_train_data.nodes, voxel_train_data.pcd, voxel_train_data.y

    nodes, pcd, target = join_trajectories(nodes), join_trajectories(pcd), join_trajectories(target)
    dataset = CustomDataset(data=pcd, targets=target)

    #####################
    # plot point of interest over time
    #####################

    fig = plt.figure(1, figsize=(100, 100))

    for i in range(1, 60):
        fig.add_subplot(10, 6, i)

        x, y = pcd[i][:,0], pcd[i][:,1]
        plt.scatter(x, y)

        x, y = target[i, 0], target[i, 1]

        plt.plot([x], [y], marker='o', markersize=10, color="red")

        i += 1

    plt.show()



