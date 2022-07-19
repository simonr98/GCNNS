import torch as T
import numpy as np
from matplotlib import pyplot as plt
from util.get_data import get_torus_data_loaders, get_model_net_data
from dataclasses import dataclass


@dataclass
class BatchData:
    data: T.Tensor
    batch: T.LongTensor


############################################################
# TESTS
############################################################

_, test_loader = get_torus_data_loaders(com=True, key='mesh', batch_size=1)

MODEL_PATH = 'torus_com_prediction_pcd_20_epochs'

i = 1

for data in test_loader:
    model = T.load(MODEL_PATH).cpu()
    model.eval()
    predicted_com = model(data).detach().numpy()

    #####################
    # plot point of interest over time
    #####################

    fig = plt.figure(1, figsize=(100, 100))


    fig.add_subplot(2, 3, i)

    ax = fig.add_subplot(2, 3, i, projection='3d')
    ax.set_title('Point cloud with noise', {'fontsize': 10})
    ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:,2], c='red')

    # x, y, z = CAMERA_COORDINATES_TORUS[0], CAMERA_COORDINATES_TORUS[1], CAMERA_COORDINATES_TORUS[2]

    plt.plot([predicted_com[0, 0]], [predicted_com[0, 1]], [predicted_com[0, 2]], marker='o', markersize=10, color="blue")
    plt.plot([data.y[0, 0]], [data.y[0, 1]], [data.y[0, 2]], marker='o', markersize=10, color="green")

    i += 1

    if i == 7:
        break


plt.show()


############################################################
