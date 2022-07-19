import torch as T
from torch import linalg as LA
from matplotlib import pyplot as plt
import numpy as np

from definitions import NUM_POINTS_POINT_CLOUD, BATCH_SIZE_TORUS
from util.get_data import get_torus_data_loaders, get_model_net_data
from dataclasses import dataclass


@dataclass
class BatchData:
    data: T.Tensor
    batch: T.LongTensor


############################################################
# TESTS
############################################################

# train_loader, test_loader = get_torus_data_loaders(com=True, key='mesh')

train_loader, test_loader = get_model_net_data(num_points_pc=NUM_POINTS_POINT_CLOUD, batch_size=BATCH_SIZE_TORUS)

MODEL_PATH = 'torus_com_prediction_mesh_10_epochs'

counter = 0

for data in train_loader:
    # model = T.load(MODEL_PATH).cpu()
    # model.eval()

    # predicted_com = model(data)
    # predicted_com = predicted_com.detach().numpy()

    def get_com(data):
        def batch(i):
            batch = np.where(data.batch.numpy() == i)[0]
            pos = data.pos[batch[0]:batch[-1], :]
            com = T.sum(pos, dim=0) / pos.shape[0]
            return com

        return T.stack([batch(i) for i in range(train_loader.batch_size)])

    com = get_com(data)

    com1 = com + 5

    print(T.sum(LA.vector_norm(com, dim=1)))


    exit()


    # --------------------------------------------------------------------------------------------------------

    #####################
    # plot point of interest over time
    #####################

    fig = plt.figure(1, figsize=(100, 100))

    for i in range(1, 8):
        batch = np.where(data.batch.numpy() == i)[0]
        pos = data.pos[batch[0]:batch[-1], :]
        com = T.sum(pos, dim=0) / pos.shape[0]

        print(com.shape)

        fig.add_subplot(3, 3, i)

        ax = fig.add_subplot(3, 3, i, projection='3d')
        ax.set_title('Point cloud with noise', {'fontsize': 10})
        ax.scatter(pos[:, 0], pos[:, 1], pos[:,2], c='red')

        plt.plot([com[0]], [com[1]], [com[2]], marker='o', markersize=30, color="green")


        # x, y, z = CAMERA_COORDINATES_TORUS[0], CAMERA_COORDINATES_TORUS[1], CAMERA_COORDINATES_TORUS[2]

        # plt.plot([predicted_com[i, 0]], [predicted_com[i, 1]], [predicted_com[i, 2]], marker='o', markersize=10,
        #          color="blue")
        #
        # plt.plot([data.y[i, 0]], [data.y[i, 1]], [data.y[i, 2]], marker='o', markersize=10, color="green")

        i += 1

    plt.show()

    break

    if counter == 5:
        break

    counter += 1

############################################################
